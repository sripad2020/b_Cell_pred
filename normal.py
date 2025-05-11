import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd
from keras.api.models import load_model



seq=input("enter the protein sequence: ")
def find_epitopes(sequence, window_size=9):
    epitopes = []
    start = []
    end = []
    for i in range(len(sequence) - window_size + 1):
        epitope = sequence[i:i + window_size]
        epitopes.append(epitope)
        start.append(i)
        end.append(i + window_size - 1)
    return epitopes
epi=find_epitopes(seq,window_size=9)


def is_valid_protein_sequence(protein_sequence):
    valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
    return set(protein_sequence) <= valid_letters

def calculate_atom_counts(peptide_sequence):
    atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
    aa_info = {
        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
    }
    for aa in peptide_sequence:
        aa = aa.upper()
        if aa in aa_info:
            atom_counts['H'] += aa_info[aa][0]
            atom_counts['C'] += aa_info[aa][1]
            atom_counts['N'] += aa_info[aa][2]
            atom_counts['O'] += aa_info[aa][3]
            atom_counts['S'] += aa_info[aa][4]
    return atom_counts

def calculate_amphipathicity(protein_sequence):
    hydrophobic_moment_scale = kd
    hydrophobic_moment = sum(hydrophobic_moment_scale[aa] for aa in protein_sequence)
    mean_hydrophobicity = hydrophobic_moment / len(protein_sequence)
    return hydrophobic_moment - mean_hydrophobicity

def calculate_physicochemical_properties(protein_sequence):
    if not is_valid_protein_sequence(protein_sequence):
        return [None] * 19

    protein_analyzer = ProteinAnalysis(protein_sequence)
    theoretical_pI = protein_analyzer.isoelectric_point()
    aliphatic_index = sum(kd[aa] for aa in protein_sequence) / len(protein_sequence)
    positive_residues = sum(protein_sequence.count(aa) for aa in ['R', 'K', 'H'])
    negative_residues = sum(protein_sequence.count(aa) for aa in ['D', 'E'])
    aromatic_count = protein_analyzer.aromaticity() * len(protein_sequence)
    polar_amino_acids = set("STNQ")
    non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    polar_count = sum(protein_sequence.count(aa) for aa in polar_amino_acids)
    nonpolar_count = sum(protein_sequence.count(aa) for aa in non_polar_amino_acids)
    amino_acid_composition = protein_analyzer.get_amino_acids_percent()
    molecular_weight = protein_analyzer.molecular_weight()
    instability_index = protein_analyzer.instability_index()
    aromaticity = protein_analyzer.aromaticity()
    helix_fraction, strand_fraction, coil_fraction = protein_analyzer.secondary_structure_fraction()
    charge_at_pH_7 = protein_analyzer.charge_at_pH(7.0)
    gravy = protein_analyzer.gravy()
    amphipathicity = calculate_amphipathicity(protein_sequence)
    molar_extinction_coefficient = protein_analyzer.molar_extinction_coefficient()

    return [theoretical_pI, aliphatic_index, positive_residues, negative_residues, aromatic_count,
            polar_count, nonpolar_count, amino_acid_composition, molecular_weight, instability_index,
            aromaticity, helix_fraction, strand_fraction, coil_fraction, charge_at_pH_7, gravy, amphipathicity,
            molar_extinction_coefficient[0], molar_extinction_coefficient[1]]

atomic_data=[]
def atomic():
    for i in range(len(epi)):
        atom_counts = calculate_atom_counts(epi[i])
        total_atoms = sum(atom_counts.values())
        atomic_dict = {
            'Sequence': epi[i],
            'H_Count': atom_counts['H'],
            'C_Count': atom_counts['C'],
            'N_Count': atom_counts['N'],
            'O_Count': atom_counts['O'],
            'S_Count': atom_counts['S'],
            'TotalAtoms_Count': total_atoms
        }
        atomic_data.append(atomic_dict)
    df=pd.DataFrame(atomic_data).to_csv('the_atomic_features.csv')
    return df



aa_percentage=[]
def aa_percentages():
    for i  in range(len(epi)):
        physicochemical_properties = calculate_physicochemical_properties(epi[i])
        aa_composition = physicochemical_properties[7]
        aa_percent_dict = {'Sequence': epi[i]}
        for aa in 'ARNDCEQGHILKMFPSTWYV':
            aa_percent_dict[aa + '_Percent'] = aa_composition.get(aa, 0)
        aa_percentage.append(aa_percent_dict)
    pd.DataFrame(aa_percentage).to_csv('aa_percentages.csv')
    return "abc"
print(aa_percentages())


phys=[]
def phsycio():
    for i in range(len(epi)):
        physicochemical_properties = calculate_physicochemical_properties(epi[i])
        physico_dict = {
            'Sequence': seq,
            'Theoretical.pI': physicochemical_properties[0],
            'Aliphatic.Index': physicochemical_properties[1],
            'Positive.Residues': physicochemical_properties[2],
            'Negative.Residues': physicochemical_properties[3],
            'Aromatic.Count': physicochemical_properties[4],
            'Polar.Count': physicochemical_properties[5],
            'Nonpolar.Count': physicochemical_properties[6],
            'Molecular.Weight': physicochemical_properties[8],
            'Instability.Index': physicochemical_properties[9],
            'Aromaticity': physicochemical_properties[10],
            'Helix.Fraction': physicochemical_properties[11],
            'Strand.Fraction': physicochemical_properties[12],
            'Coil.Fraction': physicochemical_properties[13],
            'Charge.at.pH.7.0': physicochemical_properties[14],
            'Gravy': physicochemical_properties[15],
            'Amphipathicity': physicochemical_properties[16],
            'Molar.Extinction.Coefficient_Reduced': physicochemical_properties[17],
            'Molar.Extinction.Coefficient_Oxidized': physicochemical_properties[18]
        }
        phys.append(physico_dict)
    physico_df = pd.DataFrame(phys).to_csv('phys.csv')
    return physico_df

'''data = pd.read_csv('the_atomic_features.csv')
def prot_to_num(sequence):
    if isinstance(sequence, str):
        aa_hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                         sequence.upper()]
        return sum(numerical_seq) / len(numerical_seq)
    else:
        raise TypeError("Input must be a string representing a protein sequence.")


data['epi_num'] = data['Sequence'].apply(prot_to_num)
cols = data[['H_Count','C_Count','O_Count','S_Count','TotalAtoms_Count','epi_num']]
seqs=data['Sequence']

model = load_model('DL_pred_atom.h5')

# 2. Get predictions (probabilities)
prediction = model.predict(cols.values)

# 3. Get predicted class indices (argmax)
predicted_classes = np.argmax(prediction, axis=1)

# 4. Print results with sequences (Step 2)
for i in range(len(prediction)):
    print(f"Predicted class: {predicted_classes[i]}, "
          f"Probabilities: {prediction[i]}, "
          f"Sequence: {seqs.values[i]}")'''