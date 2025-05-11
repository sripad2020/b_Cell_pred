import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd
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


def aa_percentages(epi):
    aa_percentage = []  # Initialize the list to store results
    for i in range(len(epi)):
        physicochemical_properties = calculate_physicochemical_properties(epi[i])
        if not physicochemical_properties or len(physicochemical_properties) <= 7:
            print(f"Skipping sequence due to invalid properties: {epi[i]}")
            continue
        aa_composition = physicochemical_properties[7]
        if not isinstance(aa_composition, dict):
            print(f"Skipping sequence due to missing or invalid composition: {epi[i]}")
            continue
        aa_percent_dict = {'Sequence': epi[i]}
        for aa in 'ARNDCEQGHILKMFPSTWYV':
            aa_percent_dict[aa + '_Percent'] = aa_composition.get(aa, 0)
        aa_percentage.append(aa_percent_dict)
    df = pd.DataFrame(aa_percentage)
    df.to_csv('aa_percentages.csv', index=False)

    return df
