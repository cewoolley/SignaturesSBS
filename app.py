from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set the backend to 'Agg' else crashes on mac...
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText  # Import AnchoredText
import seaborn as sns
import re
import os
import io
import numpy as np
import base64
from scipy.spatial.distance import cosine
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='app_usage.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)

# Load the data from the TSV file
data = pd.read_csv('COSMIC_v3.4_SBS_GRCh38.tsv', sep='\t', index_col=0)

# Load the experimental data from the TSV file
experimental_data = pd.read_csv('human_sbs96_filtered_v1_0.txt', sep='\t', index_col=0)

# Combine the data
combined_data = pd.concat([data, experimental_data], axis=1)

# File to store the pre computed similarity matrix
SIMILARITY_MATRIX_FILE = 'similarity_matrix.csv'


# sig groups for colour coding
clock_like = ['SBS1', 'SBS5']
apobec_aid_activity = ['SBS2', 'SBS13', 'SBS84', 'SBS85']
dna_repair_defects = ['SBS3', 'SBS6', 'SBS15', 'SBS20', 'SBS21', 'SBS26', 'SBS30', 'SBS36', 'SBS44']
tobacco_related = ['SBS4', 'SBS29', 'SBS92']
uv_related = ['SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS38']
polymerase_mutations = ['SBS10a', 'SBS10b', 'SBS10c', 'SBS10d', 'SBS14','SBS9']
treatments_and_chemo = ['SBS11', 'SBS25', 'SBS31', 'SBS32', 'SBS35', 'SBS86', 'SBS87', 'SBS99']
environmental_exposures = ['SBS22a', 'SBS22b', 'SBS24', 'SBS42', 'SBS88', 'SBS90']
ros_damage = ['SBS18']
unknown = ['SBS8', 'SBS12', 'SBS16', 'SBS17a', 'SBS17b', 'SBS19', 'SBS23', 'SBS28', 'SBS33', 'SBS34', 'SBS37', 'SBS39', 'SBS40a', 'SBS40b', 'SBS40c', 'SBS41', 'SBS89', 'SBS91', 'SBS93', 'SBS94', 'SBS96', 'SBS97', 'SBS98']
artefact = ['SBS27', 'SBS43', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55', 'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60', 'SBS95']

# Define colours
color_map = {
    'Clock-like': '#FF9999',
    'APOBEC and AID activity': '#66B2FF',
    'DNA repair defects': '#99FF99',
    'Tobacco-related': '#FFCC99',
    'UV-related': '#FF99FF',
    'Polymerase mutations': '#99FFFF',
    'Treatments and chemotherapies': '#FFFF99',
    'Environmental exposures': '#FF6666',
    'ROS damage': '#66FF66',
    'Unknown': '#CCCCCC',
    'Artefact': '#000000',
    'Experimental': '#FFFFFF'  # White background for hatching
}
# func to work out what group a sig is in
def get_signature_group(signature):
    if signature in experimental_data.columns:
        return 'Experimental'
    elif signature in clock_like:
        return 'Clock-like'
    elif signature in apobec_aid_activity:
        return 'APOBEC and AID activity'
    elif signature in dna_repair_defects:
        return 'DNA repair defects'
    elif signature in tobacco_related:
        return 'Tobacco-related'
    elif signature in uv_related:
        return 'UV-related'
    elif signature in polymerase_mutations:
        return 'Polymerase mutations'
    elif signature in treatments_and_chemo:
        return 'Treatments and chemotherapies'
    elif signature in environmental_exposures:
        return 'Environmental exposures'
    elif signature in ros_damage:
        return 'ROS damage'
    elif signature in unknown:
        return 'Unknown'
    elif signature in artefact:
        return 'Artefact'
    else:
        return 'Unknown'


def compute_similarity_matrix(data):
    n = len(data.columns)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = 1 - cosine(data.iloc[:, i], data.iloc[:, j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
    return pd.DataFrame(similarity_matrix, index=data.columns, columns=data.columns)

def get_similarity_matrix():
    global combined_data  # Add this line to access the combined_data
    if os.path.exists(SIMILARITY_MATRIX_FILE):
        print("Loading pre-computed similarity matrix from file...")
        similarity_matrix = pd.read_csv(SIMILARITY_MATRIX_FILE, index_col=0)
        # Check if all signatures are present in the loaded matrix
        if set(combined_data.columns) == set(similarity_matrix.columns):
            return similarity_matrix
        else:
            print("Existing similarity matrix is outdated. Recomputing...")
    
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(combined_data)
    print("Saving similarity matrix to file...")
    similarity_matrix.to_csv(SIMILARITY_MATRIX_FILE)
    return similarity_matrix

# Load or compute the similarity matrix
similarity_matrix = get_similarity_matrix()

def subtract_signatures(primary_sig, secondary_sig):
    subtracted_sig = data[primary_sig] - data[secondary_sig]
    return subtracted_sig

global contexts96order
contexts96order = [
    "A[C>A]A",
    "A[C>A]C",
    "A[C>A]G",
    "A[C>A]T",
    "C[C>A]A",
    "C[C>A]C",
    "C[C>A]G",
    "C[C>A]T",
    "G[C>A]A",
    "G[C>A]C",
    "G[C>A]G",
    "G[C>A]T",
    "T[C>A]A",
    "T[C>A]C",
    "T[C>A]G",
    "T[C>A]T",
    "A[C>G]A",
    "A[C>G]C",
    "A[C>G]G",
    "A[C>G]T",
    "C[C>G]A",
    "C[C>G]C",
    "C[C>G]G",
    "C[C>G]T",
    "G[C>G]A",
    "G[C>G]C",
    "G[C>G]G",
    "G[C>G]T",
    "T[C>G]A",
    "T[C>G]C",
    "T[C>G]G",
    "T[C>G]T",
    "A[C>T]A",
    "A[C>T]C",
    "A[C>T]G",
    "A[C>T]T",
    "C[C>T]A",
    "C[C>T]C",
    "C[C>T]G",
    "C[C>T]T",
    "G[C>T]A",
    "G[C>T]C",
    "G[C>T]G",
    "G[C>T]T",
    "T[C>T]A",
    "T[C>T]C",
    "T[C>T]G",
    "T[C>T]T",
    "A[T>A]A",
    "A[T>A]C",
    "A[T>A]G",
    "A[T>A]T",
    "C[T>A]A",
    "C[T>A]C",
    "C[T>A]G",
    "C[T>A]T",
    "G[T>A]A",
    "G[T>A]C",
    "G[T>A]G",
    "G[T>A]T",
    "T[T>A]A",
    "T[T>A]C",
    "T[T>A]G",
    "T[T>A]T",
    "A[T>C]A",
    "A[T>C]C",
    "A[T>C]G",
    "A[T>C]T",
    "C[T>C]A",
    "C[T>C]C",
    "C[T>C]G",
    "C[T>C]T",
    "G[T>C]A",
    "G[T>C]C",
    "G[T>C]G",
    "G[T>C]T",
    "T[T>C]A",
    "T[T>C]C",
    "T[T>C]G",
    "T[T>C]T",
    "A[T>G]A",
    "A[T>G]C",
    "A[T>G]G",
    "A[T>G]T",
    "C[T>G]A",
    "C[T>G]C",
    "C[T>G]G",
    "C[T>G]T",
    "G[T>G]A",
    "G[T>G]C",
    "G[T>G]G",
    "G[T>G]T",
    "T[T>G]A",
    "T[T>G]C",
    "T[T>G]G",
    "T[T>G]T",
]

global signatures
signatures = {
    'SBS1': 'Spontaneous deamination of 5-methylcytosine (clock-like signature)',
    'SBS2': 'Activity of APOBEC family of cytidine deaminases',
    'SBS3': 'Defective homologous recombination DNA damage repair',
    'SBS4': 'Tobacco smoking',
    'SBS5': 'Unknown (clock-like signature)',
    'SBS6': 'Defective DNA mismatch repair',
    'SBS7a': 'Ultraviolet light exposure',
    'SBS7b': 'Ultraviolet light exposure',
    'SBS7c': 'Ultraviolet light exposure',
    'SBS7d': 'Ultraviolet light exposure',
    'SBS8': 'Unknown',
    'SBS9': 'Polymerase eta somatic hypermutation activity',
    'SBS10a': 'Polymerase epsilon exonuclease domain mutations',
    'SBS10b': 'Polymerase epsilon exonuclease domain mutations',
    'SBS10c': 'Defective POLD1 proofreading',
    'SBS10d': 'Defective POLD1 proofreading',
    'SBS11': 'Temozolomide treatment',
    'SBS12': 'Unknown',
    'SBS13': 'Activity of APOBEC family of cytidine deaminases',
    'SBS14': 'Concurrent polymerase epsilon mutation and defective DNA mismatch repair',
    'SBS15': 'Defective DNA mismatch repair',
    'SBS16': 'Unknown',
    'SBS17a': 'Unknown',
    'SBS17b': 'Unknown',
    'SBS18': 'Damage by reactive oxygen species',
    'SBS19': 'Unknown',
    'SBS20': 'Concurrent POLD1 mutations and defective DNA mismatch repair',
    'SBS21': 'Defective DNA mismatch repair',
    'SBS22a': 'Aristolochic acid exposure',
    'SBS22b': 'Aristolochic acid exposure',
    'SBS23': 'Unknown',
    'SBS24': 'Aflatoxin exposure',
    'SBS25': 'Chemotherapy treatment',
    'SBS26': 'Defective DNA mismatch repair',
    'SBS28': 'Unknown',
    'SBS29': 'Tobacco chewing',
    'SBS30': 'Defective DNA base excision repair due to NTHL1 mutations',
    'SBS31': 'Platinum chemotherapy treatment',
    'SBS32': 'Azathioprine treatment',
    'SBS33': 'Unknown',
    'SBS34': 'Unknown',
    'SBS35': 'Platinum chemotherapy treatment',
    'SBS36': 'Defective DNA base excision repair due to MUTYH mutations',
    'SBS37': 'Unknown',
    'SBS38': 'Indirect effect of ultraviolet light',
    'SBS39': 'Unknown',
    'SBS40a': 'Unknown',
    'SBS40b': 'Unknown',
    'SBS40c': 'Unknown',
    'SBS41': 'Unknown',
    'SBS42': 'Haloalkane exposure',
    'SBS44': 'Defective DNA mismatch repair',
    'SBS84': 'Activity of activation-induced cytidine deaminase (AID)',
    'SBS85': 'Indirect effects of activation-induced cytidine deaminase (AID)',
    'SBS86': 'Unknown chemotherapy treatment',
    'SBS87': 'Thiopurine chemotherapy treatment',
    'SBS88': 'Colibactin exposure (E.coli bacteria carrying pks pathogenicity island)',
    'SBS89': 'Unknown',
    'SBS90': 'Duocarmycin exposure',
    'SBS91': 'Unknown',
    'SBS92': 'Tobacco smoking',
    'SBS93': 'Unknown',
    'SBS94': 'Unknown',
    'SBS96': 'Unknown',
    'SBS97': 'Unknown',
    'SBS98': 'Unknown',
    'SBS99': 'Melphalan exposure',
    "SBS27": "Artefact",
    "SBS43": "Artefact",
    "SBS45": "Artefact",
    "SBS46": "Artefact",
    "SBS47": "Artefact",
    "SBS48": "Artefact",
    "SBS49": "Artefact",
    "SBS50": "Artefact",
    "SBS51": "Artefact",
    "SBS52": "Artefact",
    "SBS53": "Artefact",
    "SBS54": "Artefact",
    "SBS55": "Artefact",
    "SBS56": "Artefact",
    "SBS57": "Artefact",
    "SBS58": "Artefact",
    "SBS59": "Artefact",
    "SBS60": "Artefact",
    "SBS95": "Artefact"
}

# Add experimental signatures
for sig in experimental_data.columns:
    if sig not in signatures:
        signatures[sig] = f"Experimental Signature {sig}"

# Create a dictionary to separate COSMIC and experimental signatures
signature_groups = {
    'COSMIC': list(data.columns),
    'Experimental': list(experimental_data.columns),
    'Combined': list(combined_data.columns)
}

def plot_single_signature(signature_name, current_data):
    signature = current_data[signature_name]
    df = pd.DataFrame({'Proportion': signature})
    df['MutationType'] = df.index.str.extract(r'\[(\w>\w)\]', expand=False)
    df = df.reindex(contexts96order)

    mutation_colors = {
        'C>T': '#FF0000',
        'C>G': '#000000',
        'T>A': '#808080',
        'T>G': '#FFC0CB',
        'T>C': '#00FF00',
        'C>A': '#0000FF'
    }

    fig, ax = plt.subplots(figsize=(20, 6))
    sns.barplot(x=df.index, y='Proportion', data=df, hue='MutationType', dodge=True, palette=mutation_colors, ax=ax)
    ax.set_xlabel('Trinucleotide Context')
    ax.set_ylabel('Proportion of Single Base Substitutions')
    ax.set_title(f'COSMIC Signature {signature_name}')
    ax.legend(title='Mutation Types', loc='upper right')
    plt.xticks(range(len(contexts96order)), contexts96order, rotation=90)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(bottom=0.2)

    labels = ax.get_xticklabels()
    for label in labels:
        mutation_type = re.search(r'\[(\w>\w)\]', label.get_text()).group(1)
        label.set_color(mutation_colors[mutation_type])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_base64



def plot_signature(signature, primary_sig, secondary_sig, title, current_data):
    # Create a DataFrame from the signature Series
    df = pd.DataFrame({'Proportion': signature})
    # Extract mutation types from the index using regular expressions
    df['MutationType'] = df.index.str.extract(r'\[(\w>\w)\]', expand=False)
    # Define colour mapping for mutation types
    mutation_colors = {
        'C>T': '#FF0000',
        'C>G': '#000000',
        'T>A': '#808080',
        'T>G': '#FFC0CB',
        'T>C': '#00FF00',
        'C>A': '#0000FF'
    }
    # Sort the DataFrame by 'MutationType'
    df['MutationType'] = pd.Categorical(df['MutationType'], categories=['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G'])
    df = df.reindex(contexts96order)

    # Calculate cosine similarity
    cosine_sim = similarity_matrix.loc[primary_sig, secondary_sig]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(20, 6))
    # Create a grouped bar plot
    sns.barplot(x=df.index, y='Proportion', data=df, hue='MutationType', dodge=True, palette=mutation_colors, ax=ax)

    ax.set_xlabel('Trinucleotide Context')
    ax.set_ylabel('Differential Proportional Activity per 3nt Context')
    ax.set_title(f"{title}\nCosine Similarity: {cosine_sim:.4f}")
    ax.legend(title='Mutation Types', loc='upper right')
    plt.xticks(range(len(contexts96order)), contexts96order, rotation=90) #force ordering to that of COSMIC site.
    plt.xticks(fontsize=8)

    # Set background colours above and below y=0 so we can easily see which is which, looks better hopefully.
    ax.axhspan(0, 1, facecolor='#34a1eb', alpha=0.03)  # Above y=0
    ax.axhspan(-1, 0, facecolor='#fcad03', alpha=0.03)  # Below y=0

    # Add text labels for primary and secondary signatures
    primary_sig_label = "More active in " + primary_sig
    secondary_sig_label = "More active in " + secondary_sig
    ax.text(0.05, 0.95, primary_sig_label, transform=ax.transAxes, fontsize=12, va='top', ha='left')
    ax.text(0.05, 0.05, secondary_sig_label, transform=ax.transAxes, fontsize=12, va='bottom', ha='left')

    # Adjust the bottom margin
    plt.subplots_adjust(bottom=0.2)
    # Colour code the axis labels
    labels = ax.get_xticklabels()
    for label in labels:
        mutation_type = re.search(r'\[(\w>\w)\]', label.get_text()).group(1)
        label.set_color(mutation_colors[mutation_type])

    # Find the top 10 (was5) points furthest from zero in either direction
    top_5_indices = df['Proportion'].abs().nlargest(10).index
    for idx in top_5_indices:
        mutation_label = idx
        position = df.index.get_loc(idx)  # Get the position of the index label
        ax.text(position, df.loc[idx, 'Proportion'], mutation_label, color='black', ha='center', va='bottom' if df.loc[idx, 'Proportion'] > 0 else 'top', rotation=90)



    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    # Encode the plot as base64 for embedding in HTML
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    # Close the figure to free up memory
    plt.close(fig)

    # Plot for the primary signature
    primary_sig_plot = plot_individual_signature(current_data[primary_sig], primary_sig, primary_sig)

    # Plot for the secondary signature
    secondary_sig_plot = plot_individual_signature(current_data[secondary_sig], secondary_sig, secondary_sig)

    return plot_base64, primary_sig_plot, secondary_sig_plot

def plot_individual_signature(signature, sig_name, title):
    # Create a DataFrame from the signature Series
    df = pd.DataFrame({'Proportion': signature})
    # Extract mutation types from the index using regular expressions
    df['MutationType'] = df.index.str.extract(r'\[(\w>\w)\]', expand=False)
    df = df.reindex(contexts96order)

    # Define colour mapping for mutation types
    mutation_colors = {
        'C>T': '#FF0000',
        'C>G': '#000000',
        'T>A': '#808080',
        'T>G': '#FFC0CB',
        'T>C': '#00FF00',
        'C>A': '#0000FF'
    }

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 3))
    # Create a grouped bar plot
    sns.barplot(x=df.index, y='Proportion', data=df, hue='MutationType', dodge=True, palette=mutation_colors, ax=ax)
    ax.set_xlabel('Trinucleotide Context')
    ax.set_ylabel('Proportion of Single Base Substitutions')
    ax.set_title(title)
    ax.legend().remove()
    plt.xticks(rotation=90)
    #plt.xticks(range(len(contexts96order)), contexts96order, rotation=90)
    plt.xticks(fontsize=6)
    # Adjust the bottom margin
    plt.subplots_adjust(bottom=0.5)
    # Colour code the axis labels
    labels = ax.get_xticklabels()
    for label in labels:
        mutation_type = re.search(r'\[(\w>\w)\]', label.get_text()).group(1)
        label.set_color(mutation_colors[mutation_type])

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    # Encode the plot as base64 for embedding in HTML
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    # Close the figure to free up memory
    plt.close(fig)
    return plot_base64

def rank_signatures_by_context(mutation_context, current_data):
    context_row = current_data.loc[mutation_context]
    sorted_row = context_row.sort_values(ascending=False)
    ranked_signatures_df = pd.DataFrame({'Proportion': sorted_row.values}, index=sorted_row.index)
    return ranked_signatures_df

def plot_ranked_signatures(ranked_signatures_df, mutation_context):
    fig, ax = plt.subplots(figsize=(20, 6))

    colors = []
    hatches = []
    for sig in ranked_signatures_df.index:
        group = get_signature_group(sig)
        colors.append(color_map[group])
        hatches.append('//' if group == 'Experimental' else '')

    bars = ax.bar(ranked_signatures_df.index, ranked_signatures_df['Proportion'], color=colors, edgecolor='black', linewidth=1)

    # Apply hatching to experimental signatures
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xlabel('Signature')
    ax.set_ylabel('Proportion')
    ax.set_title(f'COSMIC_v3.4_SBS_GRCh38 Signatures Ranked by Proportion of {mutation_context} activity')
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(bottom=0.25)

    # Get the highest signature and its aetiology
    highest_signature = ranked_signatures_df.index[0]
    highest_signature_aetiology = signatures[highest_signature]

    # Annotate the plot with an arrow and label
    highest_bar = ax.containers[0].get_children()[0]
    x_coord = highest_bar.get_x() + highest_bar.get_width() / 2
    y_coord = highest_bar.get_height()
    #ax.annotate('', xy=(x_coord, y_coord), xytext=(x_coord, y_coord + 0.05), arrowprops=dict(arrowstyle='->'))
    at = AnchoredText(f'Top Signature for {mutation_context}: {highest_signature} ({highest_signature_aetiology})', prop=dict(size=10), frameon=True, loc='upper right')
    ax.add_artist(at)

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color, hatch=('//' if group == 'Experimental' else '')) 
               for group, color in color_map.items()]
    plt.legend(handles, color_map.keys(), title="Aetiology Groups", loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    # Encode the plot as base64 for embedding in HTML
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    # Close the figure to free up memory
    plt.close(fig)
    return plot_base64

def rank_signatures_by_similarity(reference_signature, current_data):
    # Filter the similarity matrix to include only the columns present in current_data
    filtered_similarity_matrix = similarity_matrix.loc[current_data.columns, current_data.columns]
    
    similarities = filtered_similarity_matrix[reference_signature]
    ranked_signatures = similarities.sort_values(ascending=False)
    return ranked_signatures

def plot_ranked_signatures_by_similarity(ranked_signatures, reference_signature):
    fig, ax = plt.subplots(figsize=(20, 6))

    colors = []
    hatches = []
    for sig in ranked_signatures.index:
        group = get_signature_group(sig)
        colors.append(color_map[group])
        hatches.append('//' if group == 'Experimental' else '')

    bars = ax.bar(ranked_signatures.index, ranked_signatures.values, color=colors, edgecolor='black', linewidth=1)

    # Apply hatching to experimental signatures
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xlabel('Signature')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'COSMIC_v3.4_SBS_GRCh38 Signatures Ranked by Similarity to {reference_signature}')
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(bottom=0.25)

    # Annotate the plot with the highest similarity signature
    highest_signature = ranked_signatures.index[1]  # Index 1 because index 0 is the reference signature itself
    highest_similarity = ranked_signatures.iloc[1]
    highest_signature_aetiology = signatures[highest_signature]


    at = AnchoredText(f'Most similar: {highest_signature} ({highest_signature_aetiology}) (Cosine Similarity: {highest_similarity:.4f})',
                      prop=dict(size=10), frameon=True, loc='upper right')
    ax.add_artist(at)

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color, hatch=('//' if group == 'Experimental' else '')) 
               for group, color in color_map.items()]
    plt.legend(handles, color_map.keys(), title="Aetiology Groups", loc='center left', bbox_to_anchor=(1, 0.5))

    # Add line after the first point (reference signature)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_base64


@app.route('/')
def index():
    global signatures, signature_groups
    return render_template('index.html', signatures=signatures, signature_groups=signature_groups)

@app.route('/plot', methods=['POST'])
def generate_plot():
    primary_signature = request.form['primary_signature']
    secondary_signature = request.form['secondary_signature']
    signature_group = request.form['signature_group']
    
    # Log the plot request
    logging.info(f"Plot request: {primary_signature} - {secondary_signature} ({signature_group})")
    
    if signature_group == 'COSMIC':
        current_data = data
    elif signature_group == 'Experimental':
        current_data = experimental_data
    else:
        current_data = combined_data
    
    subtracted_signature = current_data[primary_signature] - current_data[secondary_signature]
    plot_base64, primary_sig_plot, secondary_sig_plot = plot_signature(subtracted_signature, primary_signature, secondary_signature, f'{primary_signature} - {secondary_signature}', current_data)
    return jsonify({'plot_base64': plot_base64, 'primary_sig_plot': primary_sig_plot, 'secondary_sig_plot': secondary_sig_plot})

@app.route('/context', methods=['POST'])
def get_context_ranking():
    mutation_context = request.form['mutation_context']
    signature_group = request.form['signature_group']
    
    # Log the context ranking request
    logging.info(f"Context ranking request: {mutation_context} ({signature_group})")
    
    if signature_group == 'COSMIC':
        current_data = data
    elif signature_group == 'Experimental':
        current_data = experimental_data
    else:
        current_data = combined_data
    
    ranked_signatures_df = rank_signatures_by_context(mutation_context, current_data)
    plot_base64 = plot_ranked_signatures(ranked_signatures_df, mutation_context)
    return jsonify({'plot_base64': plot_base64})

@app.route('/similarity', methods=['POST'])
def get_similarity_ranking():
    reference_signature = request.form['reference_signature']
    signature_group = request.form['signature_group']
    
    # Log the similarity ranking request
    logging.info(f"Similarity ranking request: {reference_signature} ({signature_group})")
    
    if signature_group == 'COSMIC':
        current_data = data
    elif signature_group == 'Experimental':
        current_data = experimental_data
    else:
        current_data = combined_data
    
    ranked_signatures = rank_signatures_by_similarity(reference_signature, current_data)
    plot_base64 = plot_ranked_signatures_by_similarity(ranked_signatures, reference_signature)
    return jsonify({'plot_base64': plot_base64})

@app.route('/single_signature', methods=['POST'])
def get_single_signature():
    signature_name = request.form['signature_name']
    signature_group = request.form['signature_group']
    
    # Log the single signature request
    logging.info(f"Single signature request: {signature_name} ({signature_group})")
    
    if signature_group == 'COSMIC':
        current_data = data
    elif signature_group == 'Experimental':
        current_data = experimental_data
    else:
        current_data = combined_data
    
    plot_base64 = plot_single_signature(signature_name, current_data)
    return jsonify({'plot_base64': plot_base64})


if __name__ == '__main__':
    app.run(debug=True)
