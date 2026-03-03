# src/ml/train/ — Model training (designed to run on Azure ML GPU compute)
#
# finetune_bert.py   → fine-tunes NbAiLab/nb-bert-base on our transaction dataset
#                      run this ON AZURE ML, not locally (needs T4/V100 GPU)
# azure_ml_job.py    → submits finetune_bert.py to Azure ML from your laptop (~2KB job spec)
#                      THIS is what you run locally — just the job submitter
# export_onnx.py     → converts trained PyTorch model → ONNX → int8 quantized
#                      run after training completes (can run locally with CPU, slow but works)
