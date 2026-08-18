import streamlit as st
import io
import pandas as pd
import base64
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import math
import peptides
from rdkit import Chem
from rdkit.Chem import Descriptors
from rasar import ra_similarity, ra_pred
import joblib
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import streamlit.components.v1 as components


inps = joblib.load("inputs.joblib")
tr_r = inps[0]
te_r = inps[1]
tr_c = inps[2]
te_c = inps[3]
samples = inps[4]
desc_df = inps[5]
ecfp4_reg = inps[6]
ecfp4_cls = inps[7]

#similarity calculation for AD
def ecfp4_calculator(smiles_list):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cal_fp = []
    for smi in smiles_list:
        mol = Chem.MolFromSequence(smi)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            cal_fp.append(list(fp))
        else:
            cal_fp.append([None] * 2048)

    columns = [f"ECFP4_{i+1}" for i in range(2048)]
    cal_fp_df = pd.DataFrame(cal_fp, index=smiles_list, columns=columns)
    return cal_fp_df

def data_sort(frame, id):#id is the index of the new data frame
    df_val = pd.DataFrame(frame.apply(lambda row: [x[1] for x in sorted(zip(frame.columns, row), 
                                                                        key=lambda x: x[1], reverse=True)], 
                                                                        axis=1).tolist(), index=id)
    
    return df_val

def ad_analysis(df1_des, df2_des):
    print(df1_des.shape, df2_des.shape)
    sim = ra_similarity(df1_des, df2_des).similarity_calculation(method="Tanimoto Coefficient")
    sort_val = data_sort(sim, id=df2_des.index)
    sort_val1 = sort_val.iloc[:,0]
    AD_category = pd.Series(
            np.where(sort_val1 > 0.5, "Inside AD", "Outside AD"),
            index=sort_val1.index
        )
    return AD_category

#quantitative prediction reliability
def reliability_analysis(df1, df2, op=False):
    """
    df1=training set whole
    df2=query set descriptors
    """
    des_li = ["SD_Activity", "g", "Avg_similarity", "CV_similarity"]
    __, ra_meas_tr, __, __ = ra_pred(df1=df1, df2=df2).weighted_prediction(method="Euclidean Distance", ctc=10, log_outp=True)
    selected_rasar_des = ra_meas_tr[des_li].copy()
    
    if op == True:
        c1 = selected_rasar_des["SD_Activity"] <= 0.75
        c2 = selected_rasar_des["g"] <= 0.40
        c3 = selected_rasar_des["Avg_similarity"] >= 0.85
        c4 = selected_rasar_des["CV_similarity"] <= 0.05
        selected_rasar_des["Reliability"] = np.select(
                [
                    c1 & c2 & c3 & c4,             # All criteria
                    c1 & (c2 | c3 | c4),           # Criterion 1 + at least one other
                    c1 | c2 | c3 | c4              # Any one criterion
                ],
                [
                    "Very good",
                    "Good",
                    "Moderate"
                ],
                default="Bad"
            )
    return selected_rasar_des

#descriptor calculation
def DDE_calculator(fastas, **kw):
    AA = kw['order'] if kw['order'] is not None else 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2,
        'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
        'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6,
        'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
    }
    encodings = []

    # Add DDE_ prefix to descriptor names
    diPeptides = ['DDE_' + aa1 + aa2 for aa1 in AA for aa2 in AA]

    header = ['Sequence'] + diPeptides
    encodings.append(header)

    # Use original amino acid pairs for calculations
    rawPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in rawPairs:
        myTM.append(
            (myCodons[pair[0]] / 61) *
            (myCodons[pair[1]] / 61)
        )

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for seq in fastas:
        sequence = re.sub('-', '', str(seq).upper())

        code = [sequence]
        tmpCode = [0] * 400

        for j in range(len(sequence) - 1):
            tmpCode[
                AADict[sequence[j]] * 20 +
                AADict[sequence[j + 1]]
            ] += 1

        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(
                myTM[j] *
                (1 - myTM[j]) /
                (len(sequence) - 1)
            )

        for j in range(len(tmpCode)):
            tmpCode[j] = (
                tmpCode[j] - myTM[j]
            ) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)
    return encodings

def calculate_descriptors(sequence, operator):
    amino_acids = list(sequence)
    amino_acids = [aa for aa in amino_acids if aa in desc_df.columns]
    if len(amino_acids) == 0:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float)
        )
    selected = desc_df[amino_acids]

    # Average
    if operator == "Avg":
        descriptor = selected.mean(axis=1)
    return descriptor

def aa_desc_calculator(sequences):
    results = []
    for seq in sequences:
        val = calculate_descriptors(seq, operator="Avg")
        row = val.to_dict()
        results.append(row)

    df = pd.DataFrame(results, index=sequences)
    df.columns = desc_df.iloc[:, 0].values.tolist()
    return df

polarity = {
    'A': 'nonpolar', 'C': 'polar', 'D': 'polar', 'E': 'polar', 'F': 'nonpolar',
    'G': 'nonpolar', 'H': 'polar', 'I': 'nonpolar', 'K': 'polar', 'L': 'nonpolar',
    'M': 'nonpolar', 'N': 'polar', 'P': 'nonpolar', 'Q': 'polar', 'R': 'polar',
    'S': 'polar', 'T': 'polar', 'V': 'nonpolar', 'W': 'nonpolar', 'Y': 'polar'
}

def calculate_polarity(sequence):

    polar_count = sum (1 for aa in sequence if polarity.get(aa) == 'polar')
    nonpolar_count = len(sequence) - polar_count
    return (polar_count / len(sequence), nonpolar_count / len(sequence))


def peptide_desc_calculator(sequences: list):
    all_desc = []

    for s in sequences:
        pep = peptides.Peptide(s)
        desc = pep.descriptors()

        # Additional descriptors
        desc["Aliphatic_index"] = pep.aliphatic_index()
        desc["Instability_index"] = pep.instability_index()
        desc["Net_charge_pH7"] = pep.charge(pH=7)
        desc["Isoelectric_point"] = pep.isoelectric_point()
        desc["Molecular_weight"] = pep.molecular_weight()
        desc["Boman_index"] = pep.boman()
        desc["GRAVY_index"] = pep.hydrophobicity()
        desc["Entropy"] = pep.entropy()

        # Add Polar / Nonpolar descriptors
        polar, nonpolar = calculate_polarity(s)
        desc["Polar_count"] = polar
        desc["Nonpolar_count"] = nonpolar
        all_desc.append(desc)

    df1 = pd.DataFrame(all_desc, index=sequences)
    return df1

def rdkit_desc_cal(sequence):
    cal_des = []

    for i in sequence:
        mol = Chem.MolFromSequence(i)
        all_descriptors_dict = Descriptors.CalcMolDescriptors(mol)
        cal_des.append(all_descriptors_dict)
    cal_des_df = pd.DataFrame(cal_des, index=sequence)
    return cal_des_df

#reg columns
reg_des = ["AvgIpc", "FUKS010101", "PEOE_VSA6", "VSA_EState3", "MEEJ810101", "EState_VSA11", "DDE_RP"]
#cls columns
cls_des = ["LAWE840101", "ST5", "ROSM880105", "KF4", "ROBB790101", "FINA910101", "PP1", "ROSM880102", 
           "RICJ880113", "MEEJ800101", "Z1", "SVGER8", "ROSM880101", "ROSM880104", "MEEJ800102", 
           "COWR900101", "ARGP820101", "DAYM780201"]

def descriptor_calculator(sequences: list, type:str):
    dde = DDE_calculator(sequences, order='ACDEFGHIKLMNPQRSTVWY')
    dde_df = pd.DataFrame(dde[1:], columns=dde[0], index=sequences)

    aa_desc = aa_desc_calculator(sequences)
    pep_desc = peptide_desc_calculator(sequences)
    rdkit_desc = rdkit_desc_cal(sequences)

    final_df = pd.concat([dde_df, aa_desc, pep_desc, rdkit_desc], axis=1)
    if type == "reg":
        df = final_df[reg_des].copy()
    if type == "cls":
        df = final_df[cls_des].copy()

    return df

# PAGE CONFIG
st.set_page_config(page_title="BitPepPred-RA", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* SIDEBAR */

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#082B52 0%,#0F4C81 100%) !important;
    min-width:280px !important;
    max-width:280px !important;
    border-right:none !important;
    box-shadow:4px 0 18px rgba(0,0,0,.15);
}

/* Remove all default sidebar spacing */
section[data-testid="stSidebar"] > div:first-child{
    padding:0 !important;
    margin:0 !important;
}

/* Sidebar container */
section[data-testid="stSidebar"] .block-container{
    padding:0 !important;
    margin:0 !important;
}

/* First element (logo) */
section[data-testid="stSidebar"] .element-container:first-child{
    margin:0 !important;
    padding:0 !important;
}

/* Sidebar logo */
section[data-testid="stSidebar"] img{
    display:block;
    margin:0 auto !important;
    padding:0 !important;
    border-radius:10px;
}

/* Sidebar text */
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Navigation */
div[role="radiogroup"] label{
    border-radius:10px;
    padding:10px 12px;
    margin-bottom:6px;
    transition:0.25s;
}

div[role="radiogroup"] label:hover{
    background:#1C5AA6 !important;
}

/* Info box */
section[data-testid="stSidebar"] [data-testid="stAlert"]{
    border-radius:12px;
    background:rgba(255,255,255,0.08);
}

/* Version */
section[data-testid="stSidebar"] .stCaption{
    text-align:center;
    color:#DCEBFF !important;
}

/*MAIN PAGE*/

[data-testid="stAppViewContainer"] {
    width: 100% !important;
}

[data-testid="stAppViewContainer"] > .main {
    width: 100% !important;
}

[data-testid="stAppViewContainer"] .block-container {
    width: 100% !important;
    max-width: none !important;
    padding-top: 1.2rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# BACKGROUND IMAGE
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
background = Path("assets/background.png")
if background.exists():
    bg = get_base64(background)
    st.markdown(
        f"""
        <style>
        .stApp{{
            background-image:url("data:image/png;base64,{bg}");
            background-size:cover;
            background-attachment:fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# slidebar logo
logo = Path("logo.png")
if logo.exists():
    logo_base64 = get_base64(logo)
    st.sidebar.markdown(
    f"""
    <div style="margin-top:-35px; text-align:center;">
        <img src="data:image/png;base64,{logo_base64}" width="230">
    </div>
    """,
    unsafe_allow_html=True,
)
else:
    st.sidebar.markdown("📿 BitPepPred-RA")

# SIDEBAR
page = st.sidebar.radio(
"Navigation", [ "🏠 Home", "🧬 Single Prediction", "📂 Batch Prediction", "📊 Source and Test Set", "📖 User Manual", "📧 Contact Us",]
)
st.sidebar.info(
    """
    **Cite Us**  
    *Pore & Roy., Computational and Structural Biotechnology Journal. 2025 Jul 17.*
    """
)

st.sidebar.markdown(
    """
    <div style="
        position:fixed;
        bottom:12px;
        width:250px;
        text-align:center;
        color:#DCEBFF;
        font-size:14px;">
        Version 1.0
    </div>
    """, unsafe_allow_html=True,
)

# HOME PAGE
if page == "🏠 Home":

    #Banner
    banner = Path("banner.png")
    if banner.exists():
        st.image(str(banner), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Start Prediction Button
    st.markdown("""
    <div style="text-align:center;margin-bottom:40px;">
        <h1 style="
            display:inline-block;
            background:linear-gradient(90deg,#1666E8,#0F56C8);
            color:white;
            border-radius:15px;
            padding:15px 30px;
            font-size:30px;
            font-weight:bold;
            box-shadow:0px 8px 20px rgba(0,0,0,.25);
        ">
        ✨ Predict Query: individually or in Batch
        </h1>
    </div>
    """, unsafe_allow_html=True)

   # Feature Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="
            background:#ECFAF2;
            border:1px solid #C8EBD7;
            border-radius:18px;
            padding:20px;
            min-height:220px;
            height:auto;
            box-sizing:border-box;
            overflow-wrap:break-word;
            word-break:normal;
            text-align:center;
            box-shadow:0 4px 12px rgba(0,0,0,.08);
        ">
            <div style="font-size:45px;">🎯</div>
            <div style="font-size:22px; color:blue;">Accurate Prediction</div>
            <div style="font-size:16px; color:black;">
                Similarity-informed methodology with high accuracy,
                based on validated peptide data.
            </div>
        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown("""
        <div style="
            background:#EEF5FF;
            border:1px solid #C9DDFE;
            border-radius:18px;
            padding:20px;
            min-height:220px;
            height:auto;
            box-sizing:border-box;
            overflow-wrap:break-word;
            word-break:normal;
            text-align:center;
            box-shadow:0 4px 12px rgba(0,0,0,.08);
        ">
            <div style="font-size:45px;">⚛️</div>
            <div style="font-size:22px; color:blue;">Read-Across Approach</div>
            <div style="font-size:16px; color:black;">
                Leverages information from similar peptides to predict
                bitterness of query peptide sequences.
            </div>
        </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown("""
        <div style="
            background:#FFF8E8;
            border:1px solid #F0D27B;
            border-radius:18px;
            padding:20px;
            min-height:220px;
            height:auto;
            box-sizing:border-box;
            overflow-wrap:break-word;
            word-break:normal;
            text-align:center;
            box-shadow:0 4px 12px rgba(0,0,0,.08);
        ">
            <div style="font-size:45px;">🚀</div>
            <div style="font-size:22px; color:blue;">Fast & Reliable</div>
            <div style="font-size:16px; color:black;">
                Get prediction results with AD status within seconds
                to support your research and development.
            </div>
        </div>
        """, unsafe_allow_html=True)


    with col4:
        st.markdown("""
        <div style="
            background:#F6F2FF;
            border:1px solid #D9C7FF;
            border-radius:18px;
            padding:20px;
            min-height:220px;
            height:auto;
            box-sizing:border-box;
            overflow-wrap:break-word;
            word-break:normal;
            text-align:center;
            box-shadow:0 4px 12px rgba(0,0,0,.08);
        ">
            <div style="font-size:45px;">📥</div>
            <div style="font-size:22px; color:blue;">Export Results</div>
            <div style="font-size:16px; color:black;">
                Download and analyze your prediction results in Excel
                format for further analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    #Footer
    st.markdown("""
    <div style="text-align:center;font-size:14px;color:#555;">
        © 2026 BitPepPred-RA &nbsp;&nbsp; | &nbsp;&nbsp;
        DTC Laboratory &nbsp;&nbsp; | &nbsp;&nbsp;
        Jadavpur University
    </div>
    """, unsafe_allow_html=True)
    
# SINGLE PREDICTION
elif page == "🧬 Single Prediction":
    st.title("🧬 Single Prediction")
    st.markdown("""
    <style>

    .section-title{
        background:linear-gradient(90deg,#0F4C81,#1666E8);
        color:white;
        padding:12px;
        border-radius:12px;
        text-align:center;
        font-size:26px;
        font-weight:bold;
        letter-spacing:1px;
        margin-bottom:15px;
    }

    .result-card{
        background:white;
        border-radius:15px;
        padding:18px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.12);
        border-left:6px solid #1666E8;
        text-align:center;
        height:120px;
    }

    .result-title{
        color:#0F4C81;
        font-size:18px;
        font-weight:bold;
    }

    .result-value{
        color:#1666E8;
        font-size:28px;
        font-weight:bold;
    }

    .stButton button{
        width:150%;
        height:60px;
        font-size:22px;
        font-weight:bold;
        border-radius:12px;
        background:linear-gradient(90deg,#1666E8,#0F4C81);
        color:white;
        border:none;
    }

    </style>
    """, unsafe_allow_html=True)
    sequence = st.text_area(
        "",
        height=100,
        placeholder="Give peptide sequence here...\nExample:GFLRKLKAAKKFAK"
    )
    col1,col2,col3 = st.columns([4.5,1,4.5])
    with col2:
        predict = st.button("🔬 Predict")
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    if predict:
        if not sequence.strip():
            st.warning("⚠️ Please enter an amino acid sequence.")
        else:
            sequence = sequence.strip()

            if any(char.islower() for char in sequence):
                st.warning("⚠️ Please enter the peptide sequence using CAPITAL letters only.")
                st.stop()

            invalid_chars = set(sequence) - valid_amino_acids

            if invalid_chars:
                st.error(
                    f"❌ Invalid amino acid character(s): "
                    f"{', '.join(sorted(invalid_chars))}"
                )
            if len(list(sequence))==1:
                st.warning("⚠️ Single amino acids are not allowed.")
            else:
                #st.success("✅ Valid amino acid sequence.")
                reg_des_s = descriptor_calculator([sequence], type="reg")
                cls_des_s = descriptor_calculator([sequence], type="cls")
                reg_pred = ra_pred(df1=tr_r, df2=reg_des_s).weighted_prediction(method="Euclidean Distance", ctc=10)
                cls_pred = ra_pred(df1=tr_c, df2=cls_des_s).weighted_prediction(method="Gaussian Kernel", ctc=10)

                ecfp_te = ecfp4_calculator([sequence])
                #ad_reg
                ad_stat_reg = ad_analysis(ecfp4_reg, ecfp_te)
                #ad cls
                ad_stat_cls = ad_analysis(ecfp4_cls, ecfp_te)

                #reg reliability
                s_reliability = reliability_analysis(tr_r, reg_des_s)


                # Dummy outputs
                if cls_pred.iloc[0] >= 0.5:
                    prediction = "Bitter"
                    bitterness_value = round(reg_pred.iloc[0], 3)
                    ad_status1 = ad_stat_cls.iloc[0]
                    ad_status2 = ad_stat_reg.iloc[0]
                else:
                    prediction = "Non Bitter"
                    bitterness_value = "N/A"
                    ad_status1 = ad_stat_cls.iloc[0]
                    ad_status2 = "N/A"
                

                # Prediction Summary
                st.markdown('<div class="section-title">Classification Prediction</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="result-card">
                    <div class="result-title">
                    Class predicted
                    </div>
                    <div class="result-value">
                    {prediction}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                    <div class="result-card">
                    <div class="result-title">
                    Classification AD Status
                    </div>
                    <div class="result-value">
                    {ad_status1}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.write("")
                
                st.markdown('<div class="section-title">Quantitative Prediction</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="result-card">
                    <div class="result-title">
                    Bitterness value
                    </div>
                    <div class="result-value">
                    {bitterness_value}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
            
                with c2:
                    st.markdown(f"""
                    <div class="result-card">
                    <div class="result-title">
                    AD Status
                    </div>
                    <div class="result-value">
                    {ad_status2}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.write("")

                # Reliability
                st.markdown(
                    '<div class="section-title">Quantitative Prediction Reliability</div>',
                    unsafe_allow_html=True
                    )

                if cls_pred.iloc[0] >= 0.5:

                    row = s_reliability.iloc[0]

                    # Values
                    sd = float(row["SD_Activity"])
                    g = float(row["g"])
                    avg_sim = float(row["Avg_similarity"])
                    cv_sim = float(row["CV_similarity"])

                    c1 = sd <= 0.75
                    c2 = g <= 0.40
                    c3 = avg_sim >= 0.85
                    c4 = cv_sim <= 0.05

                    criteria = [
                        ("SD activity", sd, "≤ 0.75", c1),
                        ("g", g, "≤ 0.40", c2),
                        ("Average similarity", avg_sim, "≥ 0.85", c3),
                        ("CV similarity", cv_sim, "≤ 0.05", c4)
                    ]

                    if c1 and c2 and c3 and c4:
                        reliability = "Very good"
                    elif c1 and (c2 or c3 or c4):
                        reliability = "Good"
                    elif c1 or c2 or c3 or c4:
                        reliability = "Moderate"
                    else:
                        reliability = "Bad"

                    rel_colors = {
                        "Very good": "#198754",
                        "Good": "#20c997",
                        "Moderate": "#f0ad00",
                        "Bad": "#dc3545"
                    }

                    rel_color = rel_colors[reliability]

                    html = """
                    <html>
                    <head>

                    <style>

                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 10px;
                        background: transparent;
                    }

                    .criteria-container {
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                        gap: 15px;
                        width: 100%;
                    }

                    .criteria-item {
                        text-align: center;
                        flex: 1;
                        min-width: 0;
                    }

                    .circle {
                        border-radius: 50%;
                        margin: 0 auto;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        color: white;
                        font-weight: bold;
                        box-shadow: 0px 4px 10px rgba(0,0,0,0.20);
                    }

                    .value {
                        font-size: 17px;
                        line-height: 20px;
                    }

                    .status {
                        font-size: 21px;
                        line-height: 22px;
                    }

                    .name {
                        margin-top: 12px;
                        font-weight: bold;
                        font-size: 13px;
                        color: #333;
                    }

                    .threshold {
                        margin-top: 5px;
                        font-size: 12px;
                        color: #666;
                    }

                    .reliability {
                        margin: 30px auto 5px auto;
                        padding: 12px 30px;
                        width: fit-content;
                        border-radius: 12px;
                        font-size: 21px;
                        font-weight: bold;
                    }

                    </style>

                    </head>

                    <body>

                    <div class="criteria-container">
                    """

                    for name, value, threshold, passed in criteria:

                        if passed:
                            color = "#198754"
                            symbol = "✓"
                        else:
                            color = "#dc3545"
                            symbol = "✗"

                        size = 85
                        html += f"""
                        <div class="criteria-item">

                            <div class="circle"
                                style="
                                    width:{size}px;
                                    height:{size}px;
                                    background-color:{color};
                                ">

                                <div class="value">
                                    {value:.3f}
                                </div>

                                <div class="status">
                                    {symbol}
                                </div>
                            </div>
                            <div class="name">
                                {name}
                            </div>
                            <div class="threshold">
                                Criterion: {threshold}
                            </div>
                        </div>
                        """


                    html += f"""
                    </div>
                    <div class="reliability"
                        style="
                            background-color: {rel_color}20;
                            color: {rel_color};
                        ">

                        Reliability: {reliability}

                    </div>
                    </body>
                    </html>
                    """
                    components.html(
                        html,
                        height=250,
                        scrolling=False
                    )
                
                # Amino Acid Distribution
                st.markdown('<div class="section-title">Amino Acid Distribution (%)</div>', unsafe_allow_html=True)
                amino_acids=list("ACDEFGHIKLMNPQRSTVWY")
                total=len(sequence)
                counter=Counter(sequence)
                aa=[]
                percent=[]
                for i in amino_acids:
                    aa.append(i)
                    if total==0:
                        percent.append(0)
                    else:
                        percent.append(round(counter.get(i,0)/total*100,2))
                df=pd.DataFrame({"Amino Acid":aa, "Percentage":percent})
                fig=px.bar(df, x="Percentage", y="Amino Acid", orientation="h",
                    text="Percentage", height=650)
                fig.update_layout(template="plotly_white", title="", xaxis_title="Percentage (%)",
                    yaxis_title="", font=dict(size=16),
                    margin=dict(l=40,r=20,t=20,b=20))
                st.plotly_chart(fig,use_container_width=True)

                # Statistics
                st.markdown('<div class="section-title">Sequence Statistics</div>', unsafe_allow_html=True)
                hydrophobic="AILMFWVPG"
                polar="STNQCY"
                charged="KRHDE"
                hydrophobic_count=sum(counter[x] for x in hydrophobic)
                polar_count=sum(counter[x] for x in polar)
                charged_count=sum(counter[x] for x in charged)
                cc1,cc2,cc3,cc4=st.columns(4)
                cc1.metric("Seq. Length",total)
                cc2.metric("Unique AA",len(counter))
                if total>0:
                    cc3.metric("Hydrophobic",f"{hydrophobic_count/total*100:.1f}%")
                    cc4.metric("Charged",f"{charged_count/total*100:.1f}%")

# BATCH PREDICTION
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("""
    <style>
    .section-title{
        background:linear-gradient(90deg,#0F4C81,#1666E8);
        color:white;
        padding:12px;
        border-radius:12px;
        text-align:center;
        font-size:26px;
        font-weight:bold;
        margin-bottom:15px;
    }

    .stButton button{
        width:100%;
        height:50px;
        font-size:22px;
        font-weight:bold;
        border-radius:12px;
        background:linear-gradient(90deg,#1666E8,#0F4C81);
        color:white;
        border:none;
    }

    </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([7,1]) #vertical_alignment="bottom")
    with col1:
        st.info(
            "📄 Upload an Excel (.xlsx) file containing peptide sequences. "
            "The sequence column should be named **Sequence**."
    )

    with col2:
        sample_df = pd.DataFrame({
            "Sequence": [
                "GFLRKLKAAKKFAK",
                "KLLKLLKKLL",
                "AAAAAAAAGGG"
            ]
        })

        sample_buffer = io.BytesIO()
        with pd.ExcelWriter(sample_buffer, engine="openpyxl") as writer:
                sample_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Sample"
                )

        sample_buffer.seek(0)
        st.download_button(
                    label=" Get sample file 📥",
                    data=sample_buffer,
                    file_name="Sample_Input.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=False
            )

    # Upload File
    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx"]
    )

    # Read Excel
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df.index = range(1, len(df) + 1)
        st.markdown("### 📄 Uploaded Dataset")
        st.dataframe(
            df,
            use_container_width=True
        )
        if "Sequence" not in df.columns:
            st.error("Excel file must contain a column named 'Sequence'.")
        else:

            # Predict Button
            if st.button("🔬 Predict All Sequences"):
                all_seq = df["Sequence"].astype(str).str.upper().tolist()
                reg_des_b = descriptor_calculator(all_seq, type="reg")
                cls_des_b = descriptor_calculator(all_seq, type="cls")
                reg_pred_b = ra_pred(df1=tr_r, df2=reg_des_b).weighted_prediction(method="Euclidean Distance", ctc=10)
                cls_pred_b = ra_pred(df1=tr_c, df2=cls_des_b).weighted_prediction(method="Gaussian Kernel", ctc=10)
                cls_pred_b = pd.Series(
                                    np.where(cls_pred_b > 0.5, "Bitter", "Non Bitter"),
                                    index=cls_pred_b.index
                                )
                ecfp_te_b = ecfp4_calculator(all_seq)
                #ad_reg
                ad_stat_reg_b = ad_analysis(ecfp4_reg, ecfp_te_b)
                #ad cls
                ad_stat_cls_b = ad_analysis(ecfp4_cls, ecfp_te_b)
                #reg reliability
                b_reliability = reliability_analysis(tr_r, reg_des_b, op=True)
                print(b_reliability)
                final_output = pd.DataFrame()
                final_output["Sequence"] = all_seq
                final_output["Predicted Class"] = cls_pred_b.values
                final_output["AD Status (Class.)"] = ad_stat_cls_b.values
                final_output["Bitterness Value (log1/T in M)"] = reg_pred_b.values
                final_output["AD Status (Quant.)"] = ad_stat_reg_b.values
                final_output["Prediction Reliability"] = b_reliability["Reliability"].values
                mask = final_output["Predicted Class"] == "Non Bitter"

                final_output.loc[mask, [
                    "Bitterness Value (log1/T in M)",
                    "AD Status (Quant.)",
                    "Prediction Reliability"
                ]] = "Not Applicable"
                # Create Output
                final_output.index = range(1, len(final_output) + 1)
                output = final_output

                # Display Output
                st.markdown("---")
                st.success("🎉 Prediction completed successfully!")
                st.markdown("---")
                st.markdown("## ✅ Prediction Results")
                st.dataframe(output, use_container_width=True)

                # Download Excel
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(
                    excel_buffer,
                    engine="openpyxl"
                ) as writer:
                    output.to_excel(
                        writer,
                        index=False,
                        sheet_name="Prediction Results"
                    )
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Download Prediction Results",
                    data=excel_buffer,
                    file_name="BitPepPred_RA_Prediction.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# SOURCE AND TEST SET
elif page == "📊 Source and Test Set":
    st.title("📊 Source and Test Set")

    st.markdown("### Classification analysis")
    st.info(
        "Dataset used as source and validation of the "
        "BitPepPred-RA classification."
    )

    # Source Dataset
    c1, c2 = st.columns(2)

    with c1:
        if st.button("👁 View Source Dataset", use_container_width=True):
            st.dataframe(
                tr_c,
                use_container_width=True,
                height=400
            )

    with c2:
        csv_tr_c = tr_c.to_csv(index=True).encode("utf-8")

        st.download_button(
            "📥 Download Source Dataset",
            data=csv_tr_c,
            file_name="classification_source_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Test Dataset
    c3, c4 = st.columns(2)

    with c3:
        if st.button("👁 View Test Dataset", use_container_width=True):
            st.dataframe(
                te_c,
                use_container_width=True,
                height=400
            )

    with c4:
        csv_te_c = te_c.to_csv(index=True).encode("utf-8")

        st.download_button(
            "📥 Download Test Dataset",
            data=csv_te_c,
            file_name="classification_test_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    st.markdown("### Quantitative prediction")
    st.info(
        "Dataset used as source and validation of the "
        "BitPepPred-RA quantitative prediction."
    )

    # Source Dataset
    c5, c6 = st.columns(2)

    with c5:
        if st.button("👁 View Source Dataset ", use_container_width=True):
            st.dataframe(
                tr_r,
                use_container_width=True,
                height=400
            )

    with c6:
        csv_tr_r = tr_r.to_csv(index=True).encode("utf-8")

        st.download_button(
            "📥 Download Source Dataset ",
            data=csv_tr_r,
            file_name="quantitative_source_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Test Dataset
    c7, c8 = st.columns(2)

    with c7:
        if st.button("👁 View Test Dataset ", use_container_width=True):
            st.dataframe(
                te_r,
                use_container_width=True,
                height=400
            )

    with c8:
        csv_te_r = te_r.to_csv(index=True).encode("utf-8")

        st.download_button(
            "📥 Download Test Dataset ",
            data=csv_te_r,
            file_name="quantitative_test_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

# User manual
elif page == "📖 User Manual":
    st.title("📖 User Manual")

    st.markdown("""
<h3 style="color:#0F4C81;">1. Single Prediction</h3>

<p>
Navigate to the <strong>Single Prediction</strong> page from the navigation panel. <strong>Enter a valid peptide sequence</strong> using the
single-letter amino acid codes in the input box and click the
<strong>Predict</strong> button. The server analyzes the submitted sequence and displays the
predicted class, Applicability Domain (AD) status, predicted value, prediction
reliability, amino acid composition, and sequence statistics.
</p>

<hr>

<h3 style="color:#0F4C81;">2. Batch Prediction</h3>

<p>
To analyze multiple peptide sequences simultaneously, prepare an <strong>Excel (.xlsx) file</strong> containing the required input format and upload 
it through the <strong>Batch Prediction</strong> page.
Click <strong>Predict</strong> to process all sequences together.
The prediction results can be <strong>viewed on the screen</strong> and <strong>downloaded</strong> for further analysis.
</p>

<hr>

<h3 style="color:#0F4C81;">3. Interpreting the Results</h3>

<p>
After prediction, the server provides a comprehensive summary of the results for each submitted peptide.
The <strong>Predicted Class</strong> indicates whether the peptide is classified as <strong>Bitter</strong> or
<strong>Non Bitter</strong>. The <strong>Predicted Value</strong> represents the quantitative bitterness value (log1/T in M) of the peptides which are only bitter for the prediction model.
The <strong>AD Status</strong> using the Tanimoto similarity, indicates whether the query peptide falls within the chemical and structural space covered by the sourse datapoints, thereby reflecting the reliability of the prediction. An outside AD prediction should be interpreted carefully.
A <strong>Prediction Reliability</strong> (Very good/Good/Moderate/Bad) is also provided to indicate the confidence of the model in its prediction.
For single-sequence predictions, the server additionally displays the amino acid composition and basic sequence statistics to facilitate further analysis.
Users are encouraged to interpret the prediction results in conjunction with experimental evidence and other biological information before drawing final conclusions.
</p>

<hr>

<h3 style="color:#0F4C81;">4. Terms of use</h3>

<p>
This expert system has been developed by the <strong>DTC Laboratory</strong> and is intended <strong>solely for research purposes</strong>. For any inconvenience related to the system or calculations, please feel free to contact the individual listed in the contact information section.
</p>
    """, unsafe_allow_html=True)

# CONTACT US
elif page == "📧 Contact Us":
    st.title("📧 Contact Us")
    st.markdown("""
    <h3 style="color:#0F4C81;">Principal Investigator</h3>
    <div style="line-height:1.7; margin-left:10px; margin-bottom:25px;">
    <span style="font-size:22px; font-weight:bold;">
    Prof. (Dr.) Kunal Roy
    </span><br>

    Drug Theoretics and Cheminformatics Laboratory<br>
    Department of Pharmaceutical Technology<br>
    Jadavpur University, Kolkata, India<br>
    <a href="mailto:kunal.roy@jadavpuruniversity.in">
    kunal.roy@jadavpuruniversity.in
    </a>

    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        """
        <h3 style="color:#0F4C81;">Developers</h3>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="line-height:1.7;">

        <span style="font-size:22px; font-weight:bold;">
        Somnath Ghosh
        </span><br>

        Drug Theoretics and Cheminformatics Laboratory<br>
        Department of Pharmaceutical Technology<br>
        Jadavpur University, Kolkata, India<br>
        <a href="mailto:gsomnath9734@gmail.com">
        gsomnath9734@gmail.com
        </a>

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="line-height:1.7;">
        
        <span style="font-size:22px; font-weight:bold;">
        Souvik Pore
        </span><br>
        
        Drug Theoretics and Cheminformatics Laboratory<br>
        Department of Pharmaceutical Technology<br>
        Jadavpur University, Kolkata, India<br>
        <a href="mailto:souvikpore123@gmail.com">
        souvikpore123@gmail.com
        </a>
        </div>
        """, unsafe_allow_html=True)
