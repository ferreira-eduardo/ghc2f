import argparse
import gc
import glob
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import prepare_inputs, EarlyStoppingRanking
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, create_sparse_matrix
from utils.catalog_eval import CatalogScorer, to_protocol_frame
from utils.irev_protocol import ITEM_CHUNK, SELECTION, Protocol, accuracy, rank_all

from model.ghc2f import GHC2F
from model.cf_autoencoder import CFAutoEncoder
from model.gated_hybrid_ae import GatedHybridCFAutoEncoder
from utils.leave_one_out_cv import loo_labels
from utils.train_model import train_model


MODEL_REGISTRY = {
    "cf_autoencoder": CFAutoEncoder,
    "gated_hybrid": GatedHybridCFAutoEncoder,
    "ghc2f": GHC2F,
}

# ---------------------------------------------------------------------------
# Layer search space
# ---------------------------------------------------------------------------
_WIDTH_FRACS = {
    "mf_like": 0.02,
    "narrow": 0.08,
    "mid": 0.20,
    "wide": 0.40,
}
_MIN_WIDTH = 32

def _width(n: int, frac: float) -> int:
    return max(int(round(n * frac)), _MIN_WIDTH)

def build_layer_options(n: int) -> dict:
    """{width tier} x {depth}, widths relative to n. Configs that collapse to the
    same layer sizes after clipping are kept once (first tier name wins)."""
    options, seen = {}, set()
    for tier, frac in _WIDTH_FRACS.items():

        w = _width(n, frac)
        for d in (1, 2, 3):
            sizes = [max(w // 2 ** i, _MIN_WIDTH) for i in range(d)]
            if tuple(sizes) in seen:
                continue
            seen.add(tuple(sizes))
            options[f"{tier}_d{d}"] = lambda n_, s=tuple(sizes): [n_, *s]
    return options


# Defaults for GHC2F-specific loss hyperparameters during Stage 1 (backbone
# search). Keys must be real GHC2F.__init__ kwargs (model/ghc2f.py).
STAGE1_GHC2F_DEFAULTS = dict(
    contrastive_target="text",
    stop_grad_cf=True,
    cl_weight=0.1,
    reg_weight=1e-5,
    gate_entropy_weight=0.0,
)

STAGE1_EPOCHS = 30
STAGE3_EPOCHS = 30  # full-data confirmation gets a real training budget
STAGE3_TOP_K = 3    # re-confirm top-K stage-1 trials on 100% data
STAGE1_SAMPLE_FRAC = 0.5
MAX_REVIEWS = 50    # per-user / per-item review cap for the text profilers

# Validation follows the iRev baselines' protocol (utils/irev_protocol.py): full train
# catalog, train history excluded, per-user positive threshold, graded nDCG@10 as the
# objective. Stored in each study and appended to the study names so trials scored with
# the old 99-sampled-negatives HR@10 are never mixed in.
PROTOCOL_TAG = f"full_catalog_{SELECTION}"
STUDY_SUFFIX = "new_opt"
VAL_COL = f"full_data_val_{SELECTION}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def resolve_data_path(dataset_name, data_path=None):
    if data_path:
        return data_path
    for cand in (f"datasets/{dataset_name}.csv", f"data/{dataset_name}.parquet",
                 f"datasets/{dataset_name}.parquet"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"No data file for {dataset_name}; pass --data_path")


def resolve_embeddings_path(dataset_name, embeddings_path=None):
    if embeddings_path:
        return embeddings_path
    cand = f"embeddings_reviews/{dataset_name}.npy"
    if os.path.exists(cand):
        return cand
    found = sorted(glob.glob(f"results/embeddings/{dataset_name}_*.npy"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(f"No (or ambiguous) embeddings for {dataset_name}: {found}; "
                            f"pass --embeddings_path")


class GatedAEOptimizer:
    def __init__(self, dataset_name, device, data_path, embeddings_path, split_path=None, fold=0,
                 sample_frac=STAGE1_SAMPLE_FRAC, random_state=42, batch_size=256, num_workers=0,
                 max_reviews=MAX_REVIEWS, eval_batch_size=512, eval_chunk=ITEM_CHUNK, patience=5):
        self.dataset_name = dataset_name
        self.device = device
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_reviews = max_reviews
        self.patience = patience
        self.eval_batch_size = eval_batch_size
        self.eval_chunk = eval_chunk

        print(f"Loading {data_path} ...")
        raw = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
        raw = raw.reset_index(drop=True)
        if "rating" not in raw.columns:
            raise ValueError(f"{data_path} has no rating column; the iRev protocol needs the original ratings")
        cols = [c for c in ("userId", "itemId", "rating", "timestamp") if c in raw.columns]
        self.full_df = raw[cols].copy()
        del raw
        self.full_df["userId"] = self.full_df["userId"].astype("int64")
        self.full_df["itemId"] = self.full_df["itemId"].astype("int64")
        self.full_df["review_idx"] = np.arange(len(self.full_df))

        # --- fold assignment: precomputed split file (row-aligned) or on the fly ---
        if split_path is not None:
            print(f"Using split file {split_path}")
            split = pd.read_csv(split_path)
            if not (len(split) == len(self.full_df)
                    and (split["row_id"].values == self.full_df["review_idx"].values).all()
                    and (split["userId"].values == self.full_df["userId"].values).all()
                    and (split["itemId"].values == self.full_df["itemId"].values).all()):
                raise ValueError(f"{split_path} is not row-aligned with {data_path}")
            self.full_df["split"] = split["split"].values
        else:
            print(f"No split file; computing LOO fold {fold} on the fly")
            self.full_df["split"] = loo_labels(self.full_df, fold)
        print(self.full_df["split"].value_counts().to_dict())

        # Protocol on the FULL fold (thresholds, catalog, histories), as in run_experiment.py.
        # Sampling below only picks which val users are scored, so the catalog -- and so
        # the difficulty of the ranking -- is the same at every sample fraction.
        self.protocol = Protocol(*(to_protocol_frame(self.full_df[self.full_df["split"] == s])
                                   for s in ("train", "val", "test")))
        print(f"Protocol: {self.protocol.describe()}")

        # --- review embeddings, row-aligned with full_df ---
        print(f"Loading embeddings {embeddings_path} ...")
        embedding = np.load(embeddings_path).astype(np.float32, copy=False)
        if len(embedding) != len(self.full_df):
            raise ValueError(f"embeddings rows ({len(embedding)}) != data rows ({len(self.full_df)})")
        self.df_text = pd.DataFrame(embedding)
        self.df_text["userId"] = self.full_df["userId"].values
        self.df_text["itemId"] = self.full_df["itemId"].values
        self.text_cols = [col for col in self.df_text.columns if str(col).isdigit()]
        self.text_dim = len(self.text_cols)

        self.TOTAL_ITEMS = int(self.full_df.itemId.max() + 1)
        self.TOTAL_USERS = int(self.full_df.userId.max() + 1)
        self.layer_options = build_layer_options(self.TOTAL_ITEMS)

        self.sample_frac = None
        self._fold_cache = None
        self._resample(sample_frac)

    def _resample(self, frac: float):
        """Sample a fraction of the evaluated users (those with a val row)
        and keep all of their rows."""
        if frac == self.sample_frac and self._fold_cache is not None:
            return
        eval_users = self.full_df.loc[self.full_df["split"] == "val", "userId"].unique()
        sampled = pd.Series(eval_users).sample(frac=frac, random_state=self.random_state)
        self.df = self.full_df[self.full_df["userId"].isin(sampled)]
        self.sample_frac = frac
        self._fold_cache = None
        print(f"Sampled {len(sampled):,}/{len(eval_users):,} evaluated users ({frac:.0%})")

    def _get_fold_cache(self):
        """Everything derived from the fold split is identical across every
        trial of a stage, so compute it once per sample."""
        if self._fold_cache is not None:
            return self._fold_cache

        train = self.df[self.df["split"] == "train"]

        # Val rows of the sampled users that the protocol can evaluate (item in the
        # train catalog, not a re-review of the user's train history).
        val_rows = self.protocol.rows["val"]
        val_rows = val_rows[val_rows["user"].isin(train["userId"].unique())]

        train_matrix = create_sparse_matrix(train, self.TOTAL_USERS, self.TOTAL_ITEMS)

        # Text the model may see = train reviews only (row-aligned via index),
        # oldest -> newest so the per-user cap keeps the most recent reviews.
        order = train.sort_values(["timestamp", "review_idx"], kind="mergesort").index \
            if "timestamp" in train.columns else train.index
        df_text_seen = self.df_text.loc[order]

        # Item corpus from train reviews only (previously it used every review of
        # every train item, including the held-out val/test reviews).
        item_ids, item_text, item_mask = prepare_inputs(df_text_seen, "itemId", self.text_cols,
                                                        max_reviews=self.max_reviews)
        # moved to the GPU once per fold, not once per trial
        item_ids, item_text, item_mask = (t.to(self.device) for t in (item_ids, item_text, item_mask))

        train_ds = RankingTrainDataset(train_matrix, df_text_seen, train, seed=self.random_state,
                                       max_user_reviews=self.max_reviews, mask_pos_in_input=True)

        self._fold_cache = dict(
            train_ds=train_ds, train_matrix=train_matrix, val_rows=val_rows,
            item_ids=item_ids, item_text=item_text, item_mask=item_mask,
        )
        print(f"Fold cache: {len(train_ds):,} train users, {len(val_rows):,} val rows "
              f"({int(val_rows['liked'].sum()):,} positive), {len(item_ids):,} corpus items")
        return self._fold_cache

    def _build_model_and_loader(self, model_cls, layer_sizes, nl_type, dropout, lr,
                                 tied_weights, extra_kwargs):
        cache = self._get_fold_cache()

        common_kwargs = dict(nl_type=nl_type, dp_drop_prob=dropout, learn_rate=lr,
                             tied_weights=tied_weights)

        if issubclass(model_cls, GatedHybridCFAutoEncoder):
            model = model_cls(
                layer_sizes=layer_sizes,
                num_users=self.TOTAL_USERS,
                num_items=self.TOTAL_ITEMS,
                text_dim=self.text_dim,
                text_latent_dim=64,
                **common_kwargs,
                **extra_kwargs,
            ).to(self.device)
            # The item profiler is trained end-to-end on the raw item corpus.
            model.set_item_corpus(cache["item_ids"], cache["item_text"], cache["item_mask"])
        else:
            model = model_cls(layer_sizes=layer_sizes, **common_kwargs).to(self.device)

        persistent = self.num_workers > 0
        train_loader = DataLoader(
            cache["train_ds"], batch_size=self.batch_size, shuffle=True, collate_fn=train_collate_fn,
            num_workers=self.num_workers, persistent_workers=persistent,
        )
        return model, train_loader

    def _validator(self, model):
        """validate(model) -> full-catalog val nDCG@10 of the sampled users."""
        cache = self._get_fold_cache()
        scorer = CatalogScorer(model, cache["train_matrix"], cache["train_ds"].user_text, self.text_dim,
                               self.protocol.items, self.eval_batch_size)

        def validate(_model):
            ranks = rank_all(scorer, cache["val_rows"], self.protocol.catalog,
                             self.protocol.history["val"], self.eval_chunk)
            metrics = accuracy(ranks, self.protocol.top_rating)
            print(f"  val ndcg@10={metrics['ndcg@10']:.4f}  recall@10={metrics['recall@10']:.4f}  "
                  f"mrr={metrics['mrr']:.4f}")
            return metrics[SELECTION]

        return validate

    def _run_and_cleanup(self, model, train_loader, num_epochs=STAGE1_EPOCHS, trial=None):
        """Trains the model. With `trial`, reports the running best val nDCG@10
        after every epoch so the pruner can stop bad trials early."""
        early_stopping = EarlyStoppingRanking(patience=self.patience, verbose=False)
        validate = self._validator(model)
        callback = None

        if trial is not None:
            state = {"best": -1.0}

            def callback(epoch, val_metric, **_):
                state["best"] = max(state["best"], val_metric)
                trial.report(state["best"], step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        try:
            best_val, _ = train_model(model, num_epochs, train_loader, None, early_stopping,
                                      epoch_callback=callback, validate=validate)
        finally:
            del model, early_stopping, validate
            gc.collect()
            torch.cuda.empty_cache()

        return best_val

    # ------------------------------------------------------------------
    # Stage 1: backbone / optimizer hyperparameters (all three models)
    # ------------------------------------------------------------------
    def objective_backbone(self, trial, model_cls, num_epochs):
        dropout = trial.suggest_float("dropout", 0.0, 0.6)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        nl_type = trial.suggest_categorical("nl_type", ["selu", "relu"])
        layer_option = trial.suggest_categorical("layers", list(self.layer_options.keys()))
        tied_weights = trial.suggest_categorical("tied_weights", [True, False])

        layer_sizes = self.layer_options[layer_option](self.TOTAL_ITEMS)
        extra_kwargs = STAGE1_GHC2F_DEFAULTS if model_cls is GHC2F else {}

        model, train_loader = self._build_model_and_loader(
            model_cls, layer_sizes, nl_type, dropout, lr, tied_weights, extra_kwargs
        )
        return self._run_and_cleanup(model, train_loader, num_epochs=num_epochs, trial=trial)

    def run_backbone_study(self, model_cls, n_trials=60, num_epochs=STAGE1_EPOCHS,
                           storage=None, study_name="stage1", restart=False):
        study = create_or_resume_study(storage, study_name, self.random_state, restart,
                                       expected_attrs={"sample_frac": self.sample_frac,
                                                       "epochs": num_epochs,
                                                       "protocol": PROTOCOL_TAG})
        study.optimize(lambda trial: self.objective_backbone(trial, model_cls, num_epochs),
                       n_trials=remaining_trials(study, n_trials))
        print("\n[Stage 1] Best backbone hyperparameters:")
        print(study.best_params)
        return study, study.best_params

    # ------------------------------------------------------------------
    # Stage 2: GHC2F loss hyperparameters (backbone fixed to Stage 1 winner)
    # ------------------------------------------------------------------
    def objective_loss_weights(self, trial, fixed_backbone, num_epochs):
        ghc2f_kwargs = dict(
            text_gamma=trial.suggest_float("text_gamma", 0.1, 0.9),
            contrastive_target=trial.suggest_categorical("contrastive_target", ["text", "collaborative"]),
            stop_grad_cf=trial.suggest_categorical("stop_grad_cf", [True, False]),
            cl_weight=trial.suggest_float("cl_weight", 1e-3, 1.0, log=True),
            reg_weight=trial.suggest_float("reg_weight", 1e-6, 1e-3, log=True),
            # gate_entropy_weight is not searched: the default fusion_mode is
            # "film", where the entropy term is a no-op (see GHC2F.gate_entropy_loss).
            gate_entropy_weight=0.0,
        )
        layer_sizes = self.layer_options[fixed_backbone["layers"]](self.TOTAL_ITEMS)

        model, train_loader = self._build_model_and_loader(
            GHC2F, layer_sizes, fixed_backbone["nl_type"], fixed_backbone["dropout"],
            fixed_backbone["lr"], fixed_backbone["tied_weights"], ghc2f_kwargs,
        )
        return self._run_and_cleanup(model, train_loader, num_epochs=num_epochs, trial=trial)

    def run_loss_weight_study(self, fixed_backbone, n_trials=50, num_epochs=STAGE1_EPOCHS,
                              storage=None, study_name="stage2", restart=False):
        # Stage 2 depends on the Stage 1 winner: if that changed since the
        # stored Stage 2 trials were run, they are stale -> restart Stage 2.
        study = create_or_resume_study(storage, study_name, self.random_state, restart,
                                       expected_attrs={"sample_frac": self.sample_frac,
                                                       "epochs": num_epochs,
                                                       "protocol": PROTOCOL_TAG,
                                                       "fixed_backbone": dict(fixed_backbone)},
                                       restart_on_mismatch=True)
        study.optimize(lambda trial: self.objective_loss_weights(trial, fixed_backbone, num_epochs),
                       n_trials=remaining_trials(study, n_trials))
        print("\n[Stage 2] Best GHC2F loss weights:")
        print(study.best_params)
        return study, study.best_params

    # ------------------------------------------------------------------
    # Stage 3: retrain the top-K Stage 1 configs on 100% of the fold.
    # ------------------------------------------------------------------
    def confirm_on_full_data(self, model_cls, candidate_params_list, extra_kwargs=None,
                             num_epochs=STAGE3_EPOCHS):
        self._resample(1.0)
        extra_kwargs = extra_kwargs or {}
        results = []
        for i, params in enumerate(candidate_params_list):
            print(f"\n[Stage 3] Confirming candidate {i + 1}/{len(candidate_params_list)} on 100% data: {params}")
            layer_sizes = self.layer_options[params["layers"]](self.TOTAL_ITEMS)
            model, train_loader = self._build_model_and_loader(
                model_cls, layer_sizes, params["nl_type"], params["dropout"], params["lr"],
                params["tied_weights"], extra_kwargs,
            )
            val_metric = self._run_and_cleanup(model, train_loader, num_epochs=num_epochs)
            results.append({**params, VAL_COL: val_metric})

        results_df = pd.DataFrame(results).sort_values(VAL_COL, ascending=False)
        print("\n[Stage 3] Full-data confirmation ranking:")
        print(results_df[["layers", "tied_weights", VAL_COL]])
        return results_df


def make_storage(url):
    """None -> in-memory. For an RDB URL (e.g. sqlite:///old_optuna.db), a
    heartbeat marks trials of a crashed/killed run as FAIL, so a resumed run
    doesn't count them as finished."""
    if url is None:
        return None
    return optuna.storages.RDBStorage(url=url, heartbeat_interval=60, grace_period=180)


def _delete_study(storage, study_name):
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"Deleted existing study {study_name!r}")
    except KeyError:
        pass


def create_or_resume_study(storage, study_name, seed, restart=False, expected_attrs=None,
                           restart_on_mismatch=False):
    """expected_attrs: settings the stored trials must have been run with
    (e.g. sample_frac). On a mismatch the study is restarted when
    restart_on_mismatch, otherwise an error asks for --restart, so trials
    from different setups are never mixed in one study."""
    expected_attrs = expected_attrs or {}

    def _create():
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
        return optuna.create_study(
            study_name=study_name, storage=storage, load_if_exists=storage is not None,
            direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=seed),
        )

    if storage is not None and restart:
        _delete_study(storage, study_name)
    study = _create()

    if study.trials:
        stored = {k: study.user_attrs.get(k) for k in expected_attrs}
        if stored != expected_attrs:
            msg = f"Study {study_name!r} was run with {stored}, now {expected_attrs}"
            if not restart_on_mismatch:
                raise ValueError(msg + ". Use --restart (or another --study_prefix) to start over.")
            print(msg + " -> restarting it")
            _delete_study(storage, study_name)
            study = _create()

    for k, v in expected_attrs.items():
        study.set_user_attr(k, v)
    # A resumed run with the same seed would replay the same random
    # suggestions as the first run -> offset the seed by the trials so far.
    study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials))
    return study


def remaining_trials(study, n_trials):
    """Trials still to run so the study ends with n_trials finished
    (COMPLETE or PRUNED) trials. FAILed ones are re-run."""
    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,
                                                   optuna.trial.TrialState.PRUNED))
    left = max(n_trials - len(done), 0)
    print(f"Study {study.study_name!r}: {len(done)} finished trials, running {left} more")
    return left


def top_k_trial_params(study, k):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    return [t.params for t in completed[:k]]


def get_stage2_sample_frac(dataset_name: str) -> float:
    """Rotten Tomatoes has the fewest evaluated users, so it gets a larger
    Stage 2 sample to keep the loss-weight comparison less noisy."""
    if "rotten" in dataset_name.lower():
        return 0.7
    return 0.6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name, e.g. rotten_tomatoes")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--splits_dir", type=str, default="splits",
                        help="Uses {splits_dir}/{dataset}_fold{fold}.csv when it exists")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Default: datasets/{name}.csv, else data/{name}.parquet")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Default: embeddings_reviews/{name}.npy, else results/embeddings/{name}_*.npy")
    parser.add_argument("--stage1_trials", type=int, default=60)
    parser.add_argument("--stage2_trials", type=int, default=50)
    parser.add_argument("--stage1_frac", type=float, default=STAGE1_SAMPLE_FRAC)
    parser.add_argument("--stage2_frac", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=STAGE1_EPOCHS, help="Epochs per Stage 1/2 trial")
    parser.add_argument("--stage2_epochs", type=int, default=None,
                        help="Epochs per Stage 2 trial. Default: --epochs")
    parser.add_argument("--stage3_epochs", type=int, default=STAGE3_EPOCHS)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience (epochs without val nDCG@10 gain) in every stage")
    parser.add_argument("--skip_stage3", action="store_true", help="Skip the full-data confirmation stage")
    parser.add_argument("--out_dir", type=str, default="grid_search/old_best_hyperparameters")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Users per batch. Keep it well below the number of evaluated users "
                             "(~3k on rotten_tomatoes) or each epoch is only 1-2 optimizer steps.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--storage", type=str, default="sqlite:///old_optuna.db",
                        help="Optuna storage URL; studies are resumed from it. Pass 'none' for in-memory.")
    parser.add_argument("--study_prefix", type=str, default=None,
                        help="Default: {dataset}_{model}_fold{fold}")
    parser.add_argument("--restart", action="store_true",
                        help="Delete this run's existing studies and start from scratch")
    parser.add_argument("--max_reviews", type=int, default=MAX_REVIEWS,
                        help="Cap on reviews per user / per item fed to the text profilers")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Users per model forward at validation")
    parser.add_argument("--eval_chunk", type=int, default=ITEM_CHUNK,
                        help="Val rows ranked per pass (each holds a rows x catalog score matrix on the GPU)")
    args = parser.parse_args()

    model_cls = MODEL_REGISTRY[args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    storage = None if args.storage.lower() == "none" else make_storage(args.storage)
    study_prefix = args.study_prefix or f"{args.dataset}_{args.model}_fold{args.fold}"
    study_prefix = f"{study_prefix}_{STUDY_SUFFIX}"  # keeps old HR@10 studies apart

    split_path = os.path.join(args.splits_dir, f"{args.dataset}_fold{args.fold}.csv")
    split_path = split_path if os.path.exists(split_path) else None

    opt = GatedAEOptimizer(
        dataset_name=args.dataset, device=device,
        data_path=resolve_data_path(args.dataset, args.data_path),
        embeddings_path=resolve_embeddings_path(args.dataset, args.embeddings_path),
        split_path=split_path, fold=args.fold, sample_frac=args.stage1_frac,
        batch_size=args.batch_size, num_workers=args.num_workers, max_reviews=args.max_reviews,
        eval_batch_size=args.eval_batch_size, eval_chunk=args.eval_chunk, patience=args.patience,
    )

    # --- Stage 1: backbone search ---
    print(f"=== Stage 1: backbone search for {args.dataset} / {args.model} "
          f"(fold {args.fold}, {args.stage1_frac:.0%}) ===")
    stage1_study, best_backbone = opt.run_backbone_study(
        model_cls, n_trials=args.stage1_trials, num_epochs=args.epochs,
        storage=storage, study_name=f"{study_prefix}_stage1", restart=args.restart,
    )

    combined_params = dict(best_backbone)
    combined_params.update(dataset=args.dataset, model=args.model, fold=args.fold,
                           split_file=split_path or "on_the_fly",
                           stage1_sample_frac=args.stage1_frac, protocol=PROTOCOL_TAG)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"{args.dataset}_{args.model}")
    stage1_study.trials_dataframe().to_csv(f"{prefix}_stage1_trials.csv", index=False)

    # --- Stage 2: loss-weight search (GHC2F only), backbone fixed ---
    ghc2f_kwargs = dict(STAGE1_GHC2F_DEFAULTS)
    if model_cls is GHC2F:
        stage2_frac = args.stage2_frac or get_stage2_sample_frac(args.dataset)
        print(f"\n=== Stage 2: loss-weight search for {args.dataset} ({stage2_frac:.0%}) ===")
        opt._resample(stage2_frac)
        stage2_study, best_loss_weights = opt.run_loss_weight_study(
            fixed_backbone=best_backbone, n_trials=args.stage2_trials,
            num_epochs=args.stage2_epochs or args.epochs,
            storage=storage, study_name=f"{study_prefix}_stage2", restart=args.restart,
        )
        ghc2f_kwargs.update(best_loss_weights)
        combined_params.update(best_loss_weights)
        combined_params["stage2_sample_frac"] = stage2_frac
        stage2_study.trials_dataframe().to_csv(f"{prefix}_stage2_trials.csv", index=False)

    # --- Stage 3: confirm top-K Stage 1 candidates on 100% of the fold ---
    if not args.skip_stage3:
        print(f"\n=== Stage 3: full-data confirmation for {args.dataset} / {args.model} ===")
        candidates = top_k_trial_params(stage1_study, STAGE3_TOP_K)
        confirmation_df = opt.confirm_on_full_data(
            model_cls, candidates, extra_kwargs=ghc2f_kwargs if model_cls is GHC2F else None,
            num_epochs=args.stage3_epochs,
        )
        confirmation_df.to_csv(f"{prefix}_stage3_confirmation.csv", index=False)
        confirmed_best = confirmation_df.iloc[0].to_dict()
        for key in ("layers", "tied_weights", "nl_type", "dropout", "lr"):
            combined_params[key] = confirmed_best[key]
        combined_params[VAL_COL] = confirmed_best[VAL_COL]

    out_path = f"{prefix}.csv"
    pd.DataFrame([combined_params]).to_csv(out_path, index=False)
    print(f"\nSaved combined hyperparameters to {out_path}")
