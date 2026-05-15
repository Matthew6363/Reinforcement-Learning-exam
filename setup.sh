# Setup the python environment

python3 -m venv QSGRM_env
source QSGRM_env/bin/activate

pip install --upgrade pip
pip install pandas numpy ipykernel nashpy pygame
pip install scikit-learn matplotlib plotly