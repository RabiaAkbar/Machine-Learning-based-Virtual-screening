## **Machine Learning-Based Virtual Screening Pipeline**

This pipeline is designed for virtual screening of chemical compounds using machine learning models. The goal is to predict active and inactive compounds based on their chemical features, perform principal component analysis (PCA), and evaluate models for screening performance.

Follow the steps below to use this pipeline.


## **Step 1:** Clone the Repository

Open your terminal or command prompt.

Run the following command to clone the repository to your local machine:

    git clone https://github.com/RabiaAkbar/Machine-Learning-based-Virtual-screening.git
 
## **Step 2:** Navigate to the Repository
Change directory to the cloned repository by running the following command:

    cd Machine-Learning-based-Virtual-screening

## **Step 3:** Install Required Dependencies
This pipeline requires Python and the following libraries:

**i-scikit-learn:** For machine learning models.

**ii-RDKit:** For chemical descriptors and molecular fingerprints.

**iii-pandas:** For data manipulation and handling CSV files.

**iv-matplotlib:**
 For visualizing the chemical space (PCA plots).

Install all necessary dependencies by running:

    pip install -r requirements.txt


## **Step 4:** Prepare Your Dataset
Create a CSV file containing your chemical compounds. Ensure the CSV file includes the following columns:

**i-compound_id:** Unique identifier for each compound.
**ii-SMILES:** The SMILES representation of the compound.
**iii-Descriptors:** Various molecular descriptors (e.g., molecular weight, logP).

Example structure of your CSV file:

    compound_id,SMILES,descriptor1,descriptor2,...,descriptorN
    1,C1=CC=CC=C1,106.12,1.0,...
    2,C1=CC(=C(C=C1)O)O,138.12,2.1,...

## **Step 5:** Run the Screening Pipeline
To execute the virtual screening process, use the following command:

    python run_pipeline.py --input your_dataset.csv --output predictions.csv

This command will:

**i-** Load your dataset.

**ii-** Split the data into training and test sets.

**iii-** Train a machine learning model on your dataset.

**iv-** Predict the activities for the compounds in your dataset.

**v-** Save the predicted results in a file called predictions.csv    

## **Step 6:** View and Analyze Results
Once the screening process is complete, open the predictions.csv file to view the predicted activities. 

The file will look something like this:
    compound_id,SMILES,predicted_activity
    1,C1=CC=CC=C1,1
    2,C1=CC(=C(C=C1)O)O,0

Analyze the predictions to identify potential drug-like compounds and use the results to guide further steps in your screening workflow.    

## **Step 7:** Visualize Chemical Space
The pipeline automatically generates PCA plots to help visualize the chemical diversity and clustering of the compounds. You can find the generated plots in the plots/ directory to explore how compounds are distributed in chemical space.

## **Step 8:** Additional Steps in the Pipeline
**1-Data Collection:** Start by collecting chemical data from public databases like PubChem, ChEMBL, and BindingDB. Also, collect decoy molecules for negative examples. The collected molecules will be used for machine learning model training and validation.

**2-Feature Calculation:** Use tools like RDKit or Open Babel to calculate molecular descriptors or chemical fingerprints from SMILES representations. These features are necessary to represent molecules numerically for machine learning.

**3-Train-Test Split:** Split your dataset into training (70%) and test (30%) sets. This allows for model training and evaluation on unseen data. You can use Python’s train_test_split function from scikit-learn for this.

**4-Chemical Space and Diversity Analysis:** Perform chemical space and diversity analysis to ensure broad coverage of chemical structures. You can use clustering or visualization tools to check the diversity of the training data.

**5-Model Generation:** Train multiple machine learning models such as:

**1-SVM:** Support Vector Machine

**2-KNN:** K-Nearest Neighbors

**3-RF:** Random Forest

**4-NB:** Naive Bayes

**5-GB:** Gradient Boosting

Use the chemical features as input for these models and train them to classify active and inactive compounds.

**6-Model Evaluation:**

Evaluate the performance of the **trained models** using the test dataset. Metrics such as the **ROC-AUC curve** can help you assess how well the model distinguishes active compounds from decoys. 

Choose the best-performing model for your virtual screening.
## **Notes**
**Data Format:** Ensure your dataset is properly formatted, with valid SMILES strings and accurate descriptors.

**Customizing the Pipeline:** If you need to modify model parameters or add new features, you can edit the **config.py** file as required.