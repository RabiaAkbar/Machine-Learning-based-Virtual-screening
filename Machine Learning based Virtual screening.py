import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, EState, QED, GraphDescriptors
from rdkit.Chem import AllChem  # For calculating Morgan fingerprints
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import matthews_corrcoef, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, EState, QED, GraphDescriptors
from rdkit.Chem import AllChem  # For calculating Morgan fingerprints
import pandas as pd
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pubchempy import get_compounds








#1-Generate molecular descriptors

def generate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Failed to convert SMILES to molecule for SMILES:", smiles)
            return [np.nan] * 38  # Return NaN values for all descriptors
        # Calculate partial charges
        AllChem.ComputeGasteigerCharges(mol)
        partial_charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]

        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Descriptors.MolLogP(mol)

        max_partial_charge = max(partial_charges)
        min_partial_charge = min(partial_charges)

        estate_values = EState.EStateIndices(mol)
        max_estate = max(estate_values)
        min_estate = min(estate_values)

        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1)
        fp_density_morgan1 = sum(morgan_fp) / mol.GetNumAtoms()

        quality_estimate = QED.qed(mol)

        num_valence_electrons = Descriptors.NumValenceElectrons(mol)

        chi0 = GraphDescriptors.Chi0(mol)
        chi3n = GraphDescriptors.Chi3n(mol)
        balaban_j = GraphDescriptors.BalabanJ(mol)

        # Additional descriptors
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = Descriptors.RingCount(mol)
        tpsa = Descriptors.TPSA(mol)
        num_atoms = mol.GetNumAtoms()
        num_heavy_atoms = Descriptors.HeavyAtomCount(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        num_rings = Descriptors.RingCount(mol)
        num_aliphatic_rings = num_rings - num_aromatic_rings
        num_saturated_rings = Descriptors.NumSaturatedRings(mol)
        num_aliphatic_carbocycles = Descriptors.NumAliphaticCarbocycles(mol)
        num_aliphatic_heterocycles = Descriptors.NumAliphaticHeterocycles(mol)
        num_aliphatic_cycles = num_aliphatic_carbocycles + num_aliphatic_heterocycles
        num_aromatic_carbocycles = Descriptors.NumAromaticCarbocycles(mol)
        num_aromatic_heterocycles = Descriptors.NumAromaticHeterocycles(mol)
        num_aromatic_cycles = num_aromatic_carbocycles + num_aromatic_heterocycles
        fraction_csp3 = Descriptors.FractionCSP3(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_nh_oh = Descriptors.NOCount(mol)
        num_nh2_nh3 = Descriptors.NHOHCount(mol)
        num_bonds = mol.GetNumBonds()

        return [mol_weight, mol_logp, max_partial_charge, min_partial_charge, max_estate, min_estate,
                fp_density_morgan1, quality_estimate, num_valence_electrons, chi0, chi3n, balaban_j,
                num_heteroatoms, num_rotatable_bonds, ring_count, tpsa,
                num_atoms, num_heavy_atoms, num_aromatic_rings, num_aliphatic_rings, num_saturated_rings,
                num_aliphatic_carbocycles, num_aliphatic_heterocycles, num_aliphatic_cycles,
                num_aromatic_carbocycles, num_aromatic_heterocycles, num_aromatic_cycles, fraction_csp3,
                num_h_acceptors, num_h_donors, num_nh_oh, num_nh2_nh3, num_bonds,
                fraction_csp3, mol_weight, num_rotatable_bonds, ring_count,
                num_rotatable_bonds]
    except Exception as e:
        print(f"Error processing SMILES: {smiles}. Error: {str(e)}")
        return [np.nan] * 33  # Return NaN values for all descriptors


#2-load target compounds dataset
# Read data

data = pd.read_csv("E6p_Data.csv")

# Generate descriptors
data['descriptors'] = data['SMILES'].apply(generate_descriptors)

# Define descriptor columns
columns = ['MolWt', 'MolLogP', 'MaxPartialCharge', 'MinPartialCharge', 'MaxEStateIndex', 'MinEStateIndex',
           'FpDensityMorgan1', 'qed', 'NumValenceElectrons', 'Chi0', 'Chi3n', 'BalabanJ',
           'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 'TPSA',
           'NumAtoms', 'NumHeavyAtoms', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticCycles',
           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticCycles', 'FractionCSP3',
           'NumHAcceptors', 'NumHDonors', 'NOCount', 'NHOHCount',
           'NumBonds',
           'FractionCSP3', 'MolWt', 'NumRotatableBonds', 'RingCount',
           'NumRotatableBonds']

# Expand descriptors into separate columns
data[columns] = pd.DataFrame(data['descriptors'].tolist(), index=data.index)

# Drop the original 'descriptors' column
data.drop(columns='descriptors', inplace=True)

# Save the data with descriptors to a CSV file
output_file = 'E6p_descriptors.csv'
data.to_csv(output_file, index=False)

print(f"Descriptors saved to {output_file}")

#3- Calculate and display statistics for each column
for column in columns:
    mean_value = data[column].mean()
    median_value = data[column].median()
    mode_value = data[column].mode()[0]
    std_value = data[column].std()

    print("Statistics for", column)
    print("Mean: ", mean_value)
    print("Median: ", median_value)
    print("Mode: ", mode_value)
    print("Standard Deviation: ", std_value)
    print()

#4-split dataset into train and test set
# Your feature column names - double check these against your DataFrame columns
features = ['MolWt', 'MolLogP', 'MaxPartialCharge', 'MinPartialCharge', 'MaxEStateIndex', 'MinEStateIndex',
           'FpDensityMorgan1', 'qed', 'NumValenceElectrons', 'Chi0', 'Chi3n', 'BalabanJ',
           'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 'TPSA',
           'NumAtoms', 'NumHeavyAtoms', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticCycles',
           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticCycles', 'FractionCSP3',
           'NumHAcceptors', 'NumHDonors', 'NOCount', 'NHOHCount',
           'NumBonds',
           'FractionCSP3', 'MolWt', 'NumRotatableBonds', 'RingCount',
           'NumRotatableBonds']

# Split the data into train and test sets (70% training, 30% testing)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Verify the column names in your train_data DataFrame
print(train_data.columns)

# Get the training features and labels
X_train = train_data[features]
y_train = train_data['Label']

# Get the testing features and labels
X_test = test_data[features]
y_test = test_data['Label']

# Save to CSV files
train_data.to_csv('Train_data.csv', index=False)
test_data.to_csv('Test_data.csv', index=False)



#5-PCA Analysis
# PCA Keep only the feature columns
features = data[['MolWt', 'MolLogP', 'MaxPartialCharge', 'MinPartialCharge', 'MaxEStateIndex', 'MinEStateIndex',
           'FpDensityMorgan1', 'qed', 'NumValenceElectrons', 'Chi0', 'Chi3n', 'BalabanJ',
           'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 'TPSA',
           'NumAtoms', 'NumHeavyAtoms', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticCycles',
           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticCycles', 'FractionCSP3',
           'NumHAcceptors', 'NumHDonors', 'NOCount', 'NHOHCount',
           'NumBonds',
           'FractionCSP3', 'MolWt', 'NumRotatableBonds', 'RingCount',
           'NumRotatableBonds']]


#6-simpleimputer

# Impute missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)
features = pd.DataFrame(features_imputed, columns=features.columns)

# Now perform PCA
pca = PCA(n_components=2)

# Perform PCA on the features
principal_components = pca.fit_transform(features)

# Convert the principal components to a DataFrame
principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])

# Add back the Label column
final_df = pd.concat([principal_df, data[['Label']]], axis = 1)
print(final_df.head())
# Calculate eigenvalues
eigenvalues = pca.explained_variance_

# Calculate the variance explained by each principal component
variance_explained = pca.explained_variance_ratio_

# Print eigenvalues and variance explained
print("Eigenvalues:", eigenvalues)
print("Variance Explained:", variance_explained)




#7-load train dataset for PCA

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('Train_data.csv')
columns_to_drop = df.columns[2:13]
df = df.drop(columns=columns_to_drop)

# Separate features (X) and labels (y)
X = df.drop(['SMILES', 'Label'], axis=1)
y = df['Label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to get principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
df_pca['Label'] = y

# Perform K-Means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PCA1', 'PCA2']])

# Creating a scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('PCA with K-Means Clustering', fontsize=20)

# Define a color map for the clusters
colors = plt.cm.get_cmap('tab10', n_clusters)

# Plot each cluster with a different color
for cluster in range(n_clusters):
    indicesToKeep = df_pca['Cluster'] == cluster
    ax.scatter(df_pca.loc[indicesToKeep, 'PCA1'],
               df_pca.loc[indicesToKeep, 'PCA2'],
               c=[colors(cluster)],
               s=50,
               label=f'Cluster {cluster}')

ax.legend()
ax.grid()

# Save the figure as SVG
fig.savefig('pca_clusters.svg', format='svg')

# Show the plot
plt.show()

# Print the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)




#8-Chemical space and diverisity Analysis

# separate the features and the labels
X = final_df[['principal component 1', 'principal component 2']]
y = final_df['Label']

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=1)

# split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the Random Forest classifier on the training set
rf_classifier.fit(X_train, y_train)


# Training Set
train = plt.figure(figsize=(8, 8))
plt.scatter(X_train[y_train==0]['principal component 1'], X_train[y_train==0]['principal component 2'], color='#ff6347', alpha=0.5, label='Inactive')
plt.scatter(X_train[y_train==1]['principal component 1'], X_train[y_train==1]['principal component 2'], color='#4682b4', alpha=0.5, label='Active')
plt.title('Chemical Space of Training Set')
plt.xlabel('Molecular Weight')
plt.ylabel('LogP')
plt.legend()
plt.grid(True)
plt.show()
#Save the figure as SVG
train.savefig('scatter_plot2.svg', format='svg')

# Test Set
test=plt.figure(figsize=(8, 8))
plt.scatter(X_test[y_test==0]['principal component 1'], X_test[y_test==0]['principal component 2'], color='#ff6347', alpha=0.5, label='Inactive')
plt.scatter(X_test[y_test==1]['principal component 1'], X_test[y_test==1]['principal component 2'], color='#4682b4', alpha=0.5, label='Active')
plt.title('Chemical Space of Test Set')
plt.xlabel('Molecular Weight')
plt.ylabel('LogP')
plt.legend()
plt.grid(True)
plt.show()
#Save the figure as SVG
test.savefig('scatter_plot3.svg', format='svg')



#9-Model training

# Scale the original features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# separate the features and the labels
X = features_scaled
y = final_df['Label']

# Apply SMOTE
#smote = SMOTE(random_state=1)
#X_resampled, y_resampled = smote.fit_resample(X, y)

# split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

models = [
    ('kNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    ('SVM', SVC(class_weight='balanced'), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
    ('RF', RandomForestClassifier(class_weight='balanced'), {'n_estimators': [3, 5, 10], 'max_depth': [None]}),
    ('NB', GaussianNB(), {}),
    ('GB', GradientBoostingClassifier(), {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 1]})
]

for name, model, params in models:
    print(f'Training model: {name}')

    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    # Cross validation
    cross_val = cross_val_score(best_model, X, y, cv=10)

    print(f'Best Parameters: {grid.best_params_}')
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:\n', cm)
    print('Classification Report:\n', report)
    print(f'Matthews Correlation Coefficient: {mcc}')
    print(f'F1 Score: {f1}')
    print(f'10-fold Cross Validation: {cross_val.mean()}')





#10-Model Evalauation

# Assume previous steps for data preprocessing and splitting are done

# Define the models with the optimal hyperparameters found from GridSearchCV
models = [
    ('kNN', KNeighborsClassifier(n_neighbors=3)),
    ('SVM', SVC(gamma='scale', probability=True)), # probability=True to enable predict_proba
    ('RF', RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=5, class_weight='balanced', random_state=1)),
    ('NB', GaussianNB())
]

# Prepare the dataframe to store results
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUC'])

# Set up the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name, model in models:
    # Train the model
    print(f'Training model: {name}')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    # Cross-validation score
    cross_val = cross_val_score(model, X, y, cv=10)

    # AUC-ROC
    y_scores_test = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    y_scores_train = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    # # Add to results DataFrame
    # results = results.append({
    #     'Model': name,
    #     'Accuracy': accuracy,
    #     'Sensitivity': sensitivity,
    #     'Specificity': specificity,
    #     'MCC': mcc,
    #     'AUC': roc_auc_test
    # }, ignore_index=True)

    # Plot ROC curves for test and train sets
    ax1.plot(fpr_test, tpr_test, label=f'{name} (AUC = {roc_auc_test:.4f})')  # Test set
    ax2.plot(fpr_train, tpr_train, label=f'{name} (AUC = {roc_auc_train:.4f})')  # Train set)

# Add common ROC curve settings
for ax in [ax1, ax2]:
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

ax1.set_title('ROC-AUC on Test Set')
ax2.set_title('ROC-AUC on Train Set')

# Display the plot
plt.show()

# Save ROC curves plot as SVG
plt.savefig('roc_curves.svg', format='svg')

# Print the results
print(results)



#11-Save model in pickel library

models = [
    ('kNN', KNeighborsClassifier(n_neighbors=3)),
    ('SVM', SVC(gamma='scale')),
    ('RF', RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=1)),
    ('NB', GaussianNB())
]

for name, model in models:
    print(f'Training model: {name}')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # If the model is Random Forest, save it
    if name == 'RF':
        with open('rf_model.pkl', 'wb') as file:
            pickle.dump(model, file)



#12- Prediction new dataset

# Updated generate_descriptors function with enhanced error handling
def generate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Failed to convert SMILES to molecule for SMILES:", smiles)
            return [np.nan] * 38  # Return NaN values for all descriptors

        # Calculate partial charges
        AllChem.ComputeGasteigerCharges(mol)
        partial_charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]

        # Continue with descriptor calculations as before
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Descriptors.MolLogP(mol)
        max_partial_charge = max(partial_charges)
        min_partial_charge = min(partial_charges)
        estate_values = EState.EStateIndices(mol)
        max_estate = max(estate_values)
        min_estate = min(estate_values)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1)
        fp_density_morgan1 = sum(morgan_fp) / mol.GetNumAtoms()
        quality_estimate = QED.qed(mol)
        num_valence_electrons = Descriptors.NumValenceElectrons(mol)
        chi0 = GraphDescriptors.Chi0(mol)
        chi3n = GraphDescriptors.Chi3n(mol)
        balaban_j = GraphDescriptors.BalabanJ(mol)
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = Descriptors.RingCount(mol)
        tpsa = Descriptors.TPSA(mol)
        num_atoms = mol.GetNumAtoms()
        num_heavy_atoms = Descriptors.HeavyAtomCount(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        num_rings = Descriptors.RingCount(mol)
        num_aliphatic_rings = num_rings - num_aromatic_rings
        num_saturated_rings = Descriptors.NumSaturatedRings(mol)
        num_aliphatic_carbocycles = Descriptors.NumAliphaticCarbocycles(mol)
        num_aliphatic_heterocycles = Descriptors.NumAliphaticHeterocycles(mol)
        num_aliphatic_cycles = num_aliphatic_carbocycles + num_aliphatic_heterocycles
        num_aromatic_carbocycles = Descriptors.NumAromaticCarbocycles(mol)
        num_aromatic_heterocycles = Descriptors.NumAromaticHeterocycles(mol)
        num_aromatic_cycles = num_aromatic_carbocycles + num_aromatic_heterocycles
        fraction_csp3 = Descriptors.FractionCSP3(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_nh_oh = Descriptors.NOCount(mol)
        num_nh2_nh3 = Descriptors.NHOHCount(mol)
        num_bonds = mol.GetNumBonds()

        return [mol_weight, mol_logp, max_partial_charge, min_partial_charge, max_estate, min_estate,
                fp_density_morgan1, quality_estimate, num_valence_electrons, chi0, chi3n, balaban_j,
                num_heteroatoms, num_rotatable_bonds, ring_count, tpsa,
                num_atoms, num_heavy_atoms, num_aromatic_rings, num_aliphatic_rings, num_saturated_rings,
                num_aliphatic_carbocycles, num_aliphatic_heterocycles, num_aliphatic_cycles,
                num_aromatic_carbocycles, num_aromatic_heterocycles, num_aromatic_cycles, fraction_csp3,
                num_h_acceptors, num_h_donors, num_nh_oh, num_nh2_nh3, num_bonds,
                fraction_csp3, mol_weight, num_rotatable_bonds, ring_count,
                num_rotatable_bonds]
    except Exception as e:
        print(f"Error processing SMILES: {smiles}. Error: {str(e)}")
        return [np.nan] * 38  # Return NaN values for all descriptors

# Load the new data
new_data = pd.read_csv("New_data.csv")

# Generate descriptors for each SMILES string in the new dataset
new_data['descriptors'] = new_data['SMILES'].apply(generate_descriptors)

# Define descriptor columns
columns = ['MolWt', 'MolLogP', 'MaxPartialCharge', 'MinPartialCharge', 'MaxEStateIndex', 'MinEStateIndex',
           'FpDensityMorgan1', 'qed', 'NumValenceElectrons', 'Chi0', 'Chi3n', 'BalabanJ',
           'NumHeteroatoms', 'NumRotatableBonds', 'RingCount', 'TPSA',
           'NumAtoms', 'NumHeavyAtoms', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticCycles',
           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticCycles', 'FractionCSP3',
           'NumHAcceptors', 'NumHDonors', 'NOCount', 'NHOHCount',
           'NumBonds',
           'FractionCSP3', 'MolWt', 'NumRotatableBonds', 'RingCount',
           'NumRotatableBonds']

# Expand descriptors into separate columns
new_data[columns] = pd.DataFrame(new_data['descriptors'].tolist(), index=new_data.index)

# Drop the original 'descriptors' column
new_data.drop(columns='descriptors', inplace=True)

# Save the new data with descriptors to a CSV file
new_data.to_csv('new_data_with_descriptors.csv', index=False)

print("Descriptors for the new data have been successfully generated and saved to 'new_data_with_descriptors.csv'.")


# 1. Load the trained Random Forest model
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# 2. Load the new dataset with descriptors
new_data_with_descriptors = pd.read_csv('new_data_with_descriptors.csv')

# 3. Select the feature columns (make sure they match the training set's features)
# Assuming 'columns' is already defined in your previous code
X_new = new_data_with_descriptors[columns]

# Handle any missing values in the new data (if applicable)
X_new.fillna(0, inplace=True)

# 4. Make predictions on the new dataset
new_predictions = rf_model.predict(X_new)

# 5. Save the predictions along with the original data
new_data_with_descriptors['Predictions'] = new_predictions
new_data_with_descriptors.to_csv('new_data_predictions.csv', index=False)

print("Predictions have been successfully made and saved to 'new_data_predictions.csv'.")




#13-predict active phytohemicals

# 1. Load the predictions along with the original data
predicted_data = pd.read_csv('new_data_predictions.csv')

# 2. Define the criteria for identifying active phytochemicals
# For example, assuming that active compounds are labeled as '1' in the Predictions column
active_phytochemical = predicted_data[predicted_data['Predictions'] == 1]

# 3. Save the active phytochemical to a new CSV file
active_phytochemical.to_csv('active_phytochemical.csv', index=False)

print(f"Active phytochemical have been identified and saved to 'active_phytochemical.csv'.")



#14-apply rule of lipinski dor drug like compounds

# 1. Load the active phytochemicals dataset
active_phytochemicals = pd.read_csv('active_phytochemical.csv')

# 2. Apply Lipinski's Rule of Five
def apply_lipinski(row):
    mw = row['MolWt']
    logp = row['MolLogP']
    hbd = row['NumHDonors']
    hba = row['NumHAcceptors']
    num_rotatable_bonds = row['NumRotatableBonds']

    if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and num_rotatable_bonds <= 10):
        return True
    else:
        return False

# 3. Filter the dataset for drug-like compounds
active_phytochemicals['DrugLike'] = active_phytochemicals.apply(apply_lipinski, axis=1)
drug_like_compounds = active_phytochemicals[active_phytochemicals['DrugLike'] == True]

# 4. Sort the drug-like compounds by a chosen metric
# Example: Sorting by Molecular Weight, you can change this to any other metric
drug_like_compounds_sorted = drug_like_compounds.sort_values(by='MolWt')

# 5. Select the top 50 drug-like compounds
drug_like_compounds = drug_like_compounds_sorted.head(50)

# 6. Save thedrug-like compounds to a new CSV file
drug_like_compounds.to_csv('drug_like_compounds.csv', index=False)

print(f"The drug-like compounds have been identified and saved to 'drug_like_compounds.csv'.")

#15-calculate correaltion matrix

# Calculate the correlation matrix
corr =drug_like_compounds[['MolWt', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'NumRotatableBonds']].corr()

# Create a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Save the figure as SVG
plt.savefig('correlation_heatmap.svg')

# Show the plot
plt.show()




#16-find pubchem id and IUPAC name of drug like compounds

# Load your CSV file containing 284 drug-like compounds
df = pd.read_csv('drug.csv')

# Limit to the first 250 entries
#df = df.head(250)

# Initialize lists to store IUPAC names and PubChem IDs
iupac_names = []
pubchem_ids = []

# Loop through the SMILES strings to fetch IUPAC names and PubChem IDs
for smiles in df['SMILES']:
    try:
        compound = get_compounds(smiles, namespace='smiles')
        if compound:
            iupac_names.append(compound[0].iupac_name)
            pubchem_ids.append(compound[0].cid)
        else:
            iupac_names.append('Not found')
            pubchem_ids.append('Not found')
    except Exception as e:
        iupac_names.append('Error')
        pubchem_ids.append('Error')

# Add the IUPAC name and PubChem ID columns to the DataFrame
df['IUPAC Name'] = iupac_names
df['PubChem ID'] = pubchem_ids

# Save the output to a new CSV file
df.to_csv('output_compounds.csv', index=False)

# Display the first few rows of the output table
print(df.head())

#ML based VS is done

print("\n" + "="*50)
print("✅ Task Completed Successfully!")
print("="*50 + "\n")


