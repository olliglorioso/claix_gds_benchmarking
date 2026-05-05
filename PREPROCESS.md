# Preprocess ImageNet dataset with WebDataset
Command:
```bash
python preprocess_data.py --mode dir2tar --input imagenet_root --output ws_imagenet.tar
tar -tf ws_imagenet.tar | head -n 10
```