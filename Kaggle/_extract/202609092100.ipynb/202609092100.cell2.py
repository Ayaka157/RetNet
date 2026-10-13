# Install user-space dependencies only.
# Do NOT reinstall torch / torch_xla / libtpu in a Kaggle TPU session.
%pip install -q "transformers<5" "accelerate>=1.0" "safetensors"
