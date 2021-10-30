set -e

pip install torchtext
python -c "from torchtext.utils import download_from_url; download_from_url('https://drive.google.com/uc?id=1_zIduJngoU5sMbmjymrUPamWgtfgp2xB&export=download', root='.')"
unzip logical_fallacy_models.zip

mv logical_fallacy_models/eletra/ saved_models/
mv logical_fallacy_models/deberta/ saved_models/
rm logical_fallacy_models.zip
