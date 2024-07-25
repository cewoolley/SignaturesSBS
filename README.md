# COSMIC Signature Analyser

A Flask web application for analysing and visualising COSMIC mutational signatures.

## Features

- Compare and subtract mutational signatures
- Rank signatures by specific mutation contexts
- Find similar signatures using cosine similarity
- Visualise individual signatures
- Interactive plots and data exploration

## Setup

1. Clone the repository
2. Install required packages:
   ```
   pip install flask pandas matplotlib seaborn numpy scipy
   ```
3. Place the `COSMIC_v3.4_SBS_GRCh38.tsv` file in the project directory
4. Run the app:
   ```
   python app.py
   ```

## Usage

1. Open a web browser and navigate to `http://localhost:5000`
2. Use the dropdown menus to select signatures for comparison
3. Explore different analysis options and view the generated plots

## Data Source

This app uses COSMIC v3.4 Single Base Substitution (SBS) signatures. For more information, visit the [COSMIC Mutational Signatures website](https://cancer.sanger.ac.uk/signatures/).

## Licence

[MIT Licence](LICENCE)