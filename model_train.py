import pickle
import argparse
import torch
from torch import nn
from darts.models import TSMixerModel

# 自訂權重損失函數: Lambda 加權 MSE
class WeightedLoss(nn.Module):
    def __init__(self, lambda_weight: float):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        # pred, target shape: [batch, output_chunk_length, target_dim]
        # 將 output_chunk_length 維度去除（=1）
        if pred.dim() == 3:
            pred = pred.squeeze(1)   # [batch, target_dim]
            target = target.squeeze(1)  # [batch, target_dim]
        # 預測的第0維是實際波動度，target第0維是真實波動度，第1維是GARCH預測值
        pred_vol = pred[:, 0]
        true_vol = target[:, 0]
        garch_vol = target[:, 1]
        loss_true = self.mse(pred_vol, true_vol)
        loss_garch = self.mse(pred_vol, garch_vol)
        return self.lambda_weight * loss_true + (1 - self.lambda_weight) * loss_garch

def main():
    parser = argparse.ArgumentParser(description="Train TSMixerModel on multiple time series.")
    parser.add_argument('--data', required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.5, help="Lambda weight for custom loss")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--log_dir', default='.', help="Directory for TensorBoard logs")
    parser.add_argument('--model_path', default='tsmixer_model.pth', help="Path to save the trained model")
    args = parser.parse_args()
    # 設定隨機種子
    torch.manual_seed(args.seed)
    # 載入資料集
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
    train_targets = dataset['train']['target']
    train_covs = dataset['train']['cov']
    val_targets = dataset.get('val', {}).get('target', None)
    val_covs = dataset.get('val', {}).get('cov', None)
    # 移除 None（如果列表中有None占位，過濾掉）
    if isinstance(train_covs, list):
        train_covs = [cov for cov in train_covs if cov is not None]
    if isinstance(val_covs, list):
        val_covs = [cov for cov in val_covs if cov is not None]
    if isinstance(val_targets, list):
        val_targets = [ts for ts in val_targets if ts is not None]
    # 構建模型
    model = TSMixerModel(
        input_chunk_length=90,
        output_chunk_length=1,
        use_static_covariates=True,
        use_reversible_instance_norm=True,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        log_tensorboard=True,
        work_dir=args.log_dir,
        # 模型名稱可選擇設定
        model_name="TSMixerModel",
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": -1 if torch.cuda.is_available() else None,
            "auto_select_gpus": True if torch.cuda.is_available() else False
        }
    )
    # 模型訓練
    if val_targets and len(val_targets) > 0:
        model.fit(train_targets, past_covariates=train_covs if train_covs and len(train_covs) > 0 else None,
                  val_series=val_targets, val_past_covariates=val_covs if val_covs and len(val_covs) > 0 else None,
                  n_epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    else:
        model.fit(train_targets, past_covariates=train_covs if train_covs and len(train_covs) > 0 else None,
                  n_epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    # 保存模型
    model.save(args.model_path)
    print(f"Model trained and saved to {args.model_path}")

if __name__ == "__main__":
    main()
