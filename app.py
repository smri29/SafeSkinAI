import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import base64
import random
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import MolToImage

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="SafeSkin AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. MOLECULE DATABASE (50+ Real Cosmetic Ingredients)
# ==============================================================================
def get_molecule_database():
    return [
        {"name": "Retinol", "role": "Anti-Aging Active", "context": "Vitamin A derivative. Highly effective but can be irritating and teratogenic in high doses.", "smiles": "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=O)C)C"},
        {"name": "Propylparaben", "role": "Preservative", "context": "Common preservative linked to potential endocrine disruption in some studies.", "smiles": "CCCOC(=O)C1=CC=C(C=C1)O"},
        {"name": "Triclosan", "role": "Antimicrobial", "context": " banned in soap by FDA; concerns over thyroid/reproductive hormone disruption.", "smiles": "C1=CC(=C(C=C1Cl)O)OC2=C(C=C(C=C2)Cl)Cl"},
        {"name": "Oxybenzone", "role": "UV Filter", "context": "Chemical sunscreen agent. Known allergen and potential endocrine disruptor.", "smiles": "COC1=CC=C(C=C1)C(=O)C2=CC=CC=C2O"},
        {"name": "Formaldehyde", "role": "Contaminant/Preservative", "context": "Known carcinogen. Often released by other preservatives (e.g., DMDM Hydantoin).", "smiles": "C=O"},
        {"name": "Glycerin", "role": "Humectant", "context": "Safe, widely used moisturizing ingredient that draws water into the skin.", "smiles": "C(C(CO)O)O"},
        {"name": "Niacinamide", "role": "Skin Conditioning", "context": "Vitamin B3. Very safe, soothing, and brightening agent.", "smiles": "C1=CC(=CN=C1)C(=O)N"},
        {"name": "Salicylic Acid", "role": "Exfoliant (BHA)", "context": "Acne treatment. Safe in low concentrations, but high doses can be toxic.", "smiles": "C1=CC=C(C(=C1)C(=O)O)O"},
        {"name": "Benzoyl Peroxide", "role": "Acne Treatment", "context": "Effective antimicrobial but can generate free radicals and irritate skin.", "smiles": "C1=CC=C(C=C1)C(=O)OOC(=O)C2=CC=CC=C2"},
        {"name": "Hydroquinone", "role": "Skin Lightener", "context": "Restricted in EU/Asia due to potential carcinogenicity and ochronosis risk.", "smiles": "C1=CC(=CC=C1O)O"},
        {"name": "Sodium Lauryl Sulfate (SLS)", "role": "Surfactant", "context": "Strong cleanser. Safe but can be irritating and strip skin oils.", "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"},
        {"name": "Ascorbic Acid", "role": "Antioxidant", "context": "Pure Vitamin C. Safe and beneficial, though unstable.", "smiles": "C(C(C1C(=C(C(=O)O1)O)O)O)O"},
        {"name": "Kojic Acid", "role": "Brightening Agent", "context": "Derived from fungi. Generally safe but can cause sensitization.", "smiles": "C1=C(OC=C(C1=O)O)CO"},
        {"name": "Alpha-Tocopherol", "role": "Vitamin E", "context": "Antioxidant and moisturizer. Very safe.", "smiles": "CC1=C(C2=C(CCC(O2)(C)CCCC(C)CCCC(C)CCCC(C)C)C(=C1O)C)C"},
        {"name": "Panthenol", "role": "Soothing Agent", "context": "Pro-Vitamin B5. Excellent safety profile, promotes healing.", "smiles": "CC(C)(CO)C(C(=O)NCCCCO)O"},
        {"name": "Allantoin", "role": "Skin Protectant", "context": "Soothing agent usually derived from comfrey plant. Safe.", "smiles": "C1(C(=O)NC(=O)N1)NC(=O)N"},
        {"name": "Phenoxyethanol", "role": "Preservative", "context": "Safe alternative to parabens, but can irritate sensitive skin.", "smiles": "C1=CC=C(C=C1)OCCO"},
        {"name": "Diazolidinyl Urea", "role": "Preservative", "context": "Formaldehyde releaser. Potential allergen and carcinogen risk.", "smiles": "C1(N(CN1)CO)NC(=O)NCO"},
        {"name": "Dibutyl Phthalate", "role": "Plasticizer", "context": "Used in nail polish. Reprotoxic; banned in EU cosmetics.", "smiles": "CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC"},
        {"name": "Toluene", "role": "Solvent", "context": "Nail polish solvent. Neurotoxic and reproductive toxin.", "smiles": "CC1=CC=CC=C1"},
        {"name": "Methylisothiazolinone", "role": "Preservative", "context": "High sensitization risk. Banned in leave-on products in EU.", "smiles": "CN1C=CC(=O)S1"},
        {"name": "Cocamidopropyl Betaine", "role": "Surfactant", "context": "Mild cleanser, but impurities can cause allergies.", "smiles": "CCCCCCCCCCCC(=O)NCC[N+](C)(C)CC(=O)[O-]"},
        {"name": "Lactic Acid", "role": "AHA Exfoliant", "context": "Gentle exfoliant and humectant. Safe.", "smiles": "CC(C(=O)O)O"},
        {"name": "Glycolic Acid", "role": "AHA Exfoliant", "context": "Effective exfoliant. Can increase sun sensitivity.", "smiles": "C(C(=O)O)O"},
        {"name": "Squalane", "role": "Emollient", "context": "Skin-identical moisturizer. Very safe and stable.", "smiles": "CC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)C"},
        {"name": "Resorcinol", "role": "Hair Dye/Acne", "context": "Endocrine disruptor and irritant. Restricted use.", "smiles": "C1=CC(=CC(=C1)O)O"},
        {"name": "Butylated Hydroxyanisole (BHA)", "role": "Antioxidant", "context": "Preservative. Reasonably anticipated human carcinogen.", "smiles": "COC1=C(C=C(C=C1)C(C)(C)C)O"},
        {"name": "Homosalate", "role": "UV Filter", "context": "Sunscreen active. Potential endocrine disruptor.", "smiles": "CC1=CC(=O)C=C(C1(C)C)OC(=O)C2=CC=CC=C2"},
        {"name": "Octinoxate", "role": "UV Filter", "context": "Sunscreen active. Linked to thyroid and reproductive effects.", "smiles": "COC1=CC=C(C=C1)C=CC(=O)OCC(CC)CCCC"},
        {"name": "Avobenzone", "role": "UV Filter", "context": "UVA absorber. Generally safe but unstable in light.", "smiles": "CC(C)(C)C1=CC=C(C=C1)C(=O)CC(=O)C2=CC=C(C=C2)OC"},
        {"name": "Ceramide NP", "role": "Skin Barrier Repair", "context": "Lipid found naturally in skin. Excellent safety.", "smiles": "CCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@H](O)C=CCCCCCCCCCCCCC"},
        {"name": "Hyaluronic Acid", "role": "Humectant", "context": "Holds 1000x weight in water. Safe.", "smiles": "C(C(C1C(C(C(O1)OC2C(C(C(O2)CO)O)O)NC(=O)C)O)O)(=O)O"}, # Simplified unit
        {"name": "Azelaic Acid", "role": "Anti-inflammatory", "context": "Treats rosacea and acne. Safe.", "smiles": "C(CCCC(=O)O)CCCC(=O)O"},
        {"name": "Bakuchiol", "role": "Retinol Alternative", "context": "Plant-based anti-aging. Generally less irritating than retinol.", "smiles": "CC(=CCC/C(=C/C1=CC=C(C=C1)O)/C)C(C)C=C"},
        {"name": "Bisphenol A (BPA)", "role": "Packaging Contaminant", "context": "Endocrine disruptor. Leaches from plastics.", "smiles": "CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O"},
        {"name": "Chlorphenesin", "role": "Preservative", "context": "Synthetic preservative. Can cause dermatitis.", "smiles": "C1=CC=C(C=C1)OCC(CO)O.Cl"},
        {"name": "Dimethicone", "role": "Silicone", "context": "Forms barrier, smooths texture. Safe, large molecule.", "smiles": "C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C"},
        {"name": "Ethanol (Alcohol Denat)", "role": "Solvent/Astringent", "context": "Drying/penetration enhancer. Safe but can irritate.", "smiles": "CCO"},
        {"name": "Limonene", "role": "Fragrance", "context": "Citrus scent. Common allergen/sensitizer upon oxidation.", "smiles": "CC1=CCC(CC1)C(=C)C"},
        {"name": "Linalool", "role": "Fragrance", "context": "Floral scent. Common allergen.", "smiles": "CC(=CCCC(C)(C=C)O)C"}
    ]

# ==============================================================================
# 3. CUSTOM CSS (PREMIUM GLASSMORPHISM & ANIMATIONS)
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* IMPORT FONT */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
        
        /* GLOBAL SETTINGS */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* BACKGROUND - Deep Premium Gradient */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #1a2a33 0%, #0f171e 100%);
            color: #ffffff;
        }

        /* CARD STYLING */
        .glass-container {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }

        /* MOLECULE INFO CARD (New) */
        .info-card {
            background: linear-gradient(135deg, rgba(0, 201, 255, 0.1) 0%, rgba(0, 201, 255, 0.02) 100%);
            border-left: 4px solid #00C9FF;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .info-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 5px;
        }
        .info-role {
            font-size: 0.9rem;
            color: #00C9FF;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .info-desc {
            font-size: 0.9rem;
            color: #a8dadc;
            font-style: italic;
        }

        /* TITLES & HEADERS */
        .main-header {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            line-height: 1.2;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #8899a6;
            margin-bottom: 30px;
        }
        .section-title {
            color: #00C9FF;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 5px;
        }

        /* METRIC CARDS */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 201, 255, 0.3);
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #8899a6;
            text-transform: uppercase;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #0b1116;
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        /* BUTTONS */
        div.stButton > button {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            border: none;
            color: #0f171e;
            font-weight: 700;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            transition: all 0.3s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(0, 201, 255, 0.4);
            color: #0f171e;
        }
        
        /* FOOTER */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: rgba(11, 17, 22, 0.95);
            backdrop-filter: blur(10px);
            color: #8899a6;
            text-align: center;
            padding: 12px;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.05);
            z-index: 9999;
        }
        .footer a {
            color: #00C9FF;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s;
        }
        .footer a:hover {
            color: #92FE9D;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def render_logo(size=80):
    svg = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#00C9FF;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#92FE9D;stop-opacity:1" />
            </linearGradient>
        </defs>
        <path d="M50 10 L90 25 V50 C90 75 50 95 50 95 C50 95 10 75 10 50 V25 L50 10 Z" fill="none" stroke="url(#grad1)" stroke-width="4"/>
        <path d="M35 40 Q50 25 65 40 T35 70" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/>
        <circle cx="65" cy="40" r="5" fill="#00C9FF" />
    </svg>
    """
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')

@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts['cancer_model'] = joblib.load('cancer_model.pkl')
        artifacts['repro_model'] = joblib.load('repro_model.pkl')
        artifacts['scaler'] = joblib.load('scaler.pkl')
        with open('app_metadata.json', 'r') as f:
            artifacts['metadata'] = json.load(f)
        with open('model_stats.json', 'r') as f:
            artifacts['stats'] = json.load(f)
    except FileNotFoundError:
        return None
    return artifacts

def calc_descriptors(mol):
    if mol is None: return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HDonors': Lipinski.NumHDonors(mol),
        'HAcceptors': Lipinski.NumHAcceptors(mol),
        'Rotatable': Descriptors.NumRotatableBonds(mol),
        'Violations': sum([
            1 if Lipinski.NumHDonors(mol) > 5 else 0,
            1 if Lipinski.NumHAcceptors(mol) > 10 else 0,
            1 if Descriptors.MolWt(mol) > 500 else 0,
            1 if Descriptors.MolLogP(mol) > 5 else 0
        ]),
        # Extra descriptors for training vector
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'RingCount': Descriptors.RingCount(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'HallKierAlpha': Descriptors.HallKierAlpha(mol),
        'Kappa1': Descriptors.Kappa1(mol),
        'Kappa2': Descriptors.Kappa2(mol),
        'Kappa3': Descriptors.Kappa3(mol),
        'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'MolMR': Descriptors.MolMR(mol)
    }

def process_input(smiles, artifacts):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, "Invalid SMILES string"
    
    desc = calc_descriptors(mol)
    # Map to training feature order manually to ensure safety
    desc_values = [
        desc['MolWt'], desc['LogP'], desc['TPSA'], desc['HDonors'], desc['HAcceptors'], desc['Rotatable'],
        desc['NumAromaticRings'], desc['NumAliphaticRings'], desc['NumSaturatedRings'], desc['RingCount'],
        desc['NumHeteroatoms'], desc['FractionCSP3'], desc['BertzCT'], desc['HallKierAlpha'],
        desc['Kappa1'], desc['Kappa2'], desc['Kappa3'], desc['NumRadicalElectrons'],
        desc['NumValenceElectrons'], desc['MolMR'], desc['Violations']
    ]
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    feature_vector = np.array(desc_values + list(fp)).reshape(1, -1)
    scaled_vector = artifacts['scaler'].transform(feature_vector)
    return scaled_vector, mol

# ==============================================================================
# 5. CHART GENERATORS
# ==============================================================================
def plot_gauge(value, title, color_hex):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        title = {'text': title, 'font': {'size': 14, 'color': "white"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_hex},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
        },
        number = {'suffix': "%", 'font': {'color': "white"}}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def plot_radar(desc):
    categories = ['MolWt', 'LogP', 'H-Donors', 'H-Acceptors', 'Rotatable', 'TPSA']
    values = [
        min(desc['MolWt']/500, 1),
        min(max(desc['LogP'], 0)/5, 1), 
        min(desc['HDonors']/5, 1),
        min(desc['HAcceptors']/10, 1),
        min(desc['Rotatable']/10, 1),
        min(desc['TPSA']/150, 1)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 201, 255, 0.3)',
        line_color='#00C9FF'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        font=dict(color='white')
    )
    return fig

# ==============================================================================
# 6. UI LAYOUT & LOGIC
# ==============================================================================
artifacts = load_artifacts()
logo_img = render_logo(90)
logo_small = render_logo(50)
mol_db = get_molecule_database()

# Session State for Random Molecule
if 'surprise_idx' not in st.session_state:
    st.session_state['surprise_idx'] = random.randint(0, len(mol_db)-1)

def next_surprise():
    st.session_state['surprise_idx'] = random.randint(0, len(mol_db)-1)

# --- HEADER ---
c1, c2 = st.columns([1, 8])
with c1:
    st.markdown(f'<div style="padding-top:10px"><img src="data:image/svg+xml;base64,{logo_img}" width="100%"></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<h1 class="main-header">SafeSkin AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">The Premium Standard for Cosmetic Safety Intelligence</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f'<div style="text-align: center; margin-bottom: 20px;"><img src="data:image/svg+xml;base64,{logo_small}"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Analysis Setup</div>', unsafe_allow_html=True)
    
    input_type = st.radio("Input Source", ["Pre-set Molecule", "Custom SMILES", "Surprise Me!"], label_visibility="collapsed")
    
    smiles_input = ""
    mol_info = None
    
    if input_type == "Custom SMILES":
        smiles_input = st.text_area("Enter SMILES:", height=100, placeholder="C1=CC=C...")
    
    elif input_type == "Surprise Me!":
        mol_data = mol_db[st.session_state['surprise_idx']]
        smiles_input = mol_data['smiles']
        mol_info = mol_data # Store info to display later
        
        st.markdown(f"**Selected:** {mol_data['name']}")
        st.caption(f"Role: {mol_data['role']}")
        
        if st.button("🔄 Shuffle"):
            next_surprise()
            st.rerun()

    else:
        # Simple list for manual selection if not surprising
        names = [m['name'] for m in mol_db]
        selected = st.selectbox("Select Molecule:", names)
        # Find the full object
        mol_data = next(item for item in mol_db if item["name"] == selected)
        smiles_input = mol_data['smiles']
        mol_info = mol_data

    st.markdown("---")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        run_btn = st.button("RUN SCAN")
    with col_b2:
        if st.button("RESET"):
            st.rerun()

    st.markdown("---")
    st.caption("v2.0 Edition")

# --- MAIN CONTENT ---
if run_btn and smiles_input and artifacts:
    
    with st.spinner("🔬 Deconstructing Molecular Signature..."):
        features, mol_or_err = process_input(smiles_input, artifacts)
    
    if features is None:
        st.error(f"Analysis Error: {mol_or_err}")
    else:
        desc = calc_descriptors(mol_or_err)
        
        # Probabilities
        cancer_prob = float(artifacts['cancer_model'].predict_proba(features)[0][1])
        repro_prob = float(artifacts['repro_model'].predict_proba(features)[0][1])

        # --- MOLECULE INFO CARD ---
        if mol_info:
            st.markdown(f"""
            <div class="glass-container info-card">
                <div class="info-title">{mol_info['name']}</div>
                <div class="info-role">{mol_info['role']}</div>
                <div class="info-desc">"{mol_info['context']}"</div>
            </div>
            """, unsafe_allow_html=True)

        # --- ROW 1: STRUCTURE & RISKS ---
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown('<div class="section-title">Structure</div>', unsafe_allow_html=True)
            img = MolToImage(mol_or_err, size=(280, 280))
            st.image(img, use_container_width=True)
            if desc['Violations'] == 0:
                st.success("✅ Drug-Like")
            else:
                st.warning(f"⚠️ {desc['Violations']} Lipinski Violations")

        with col2:
            st.markdown('<div class="section-title">Carcinogenicity</div>', unsafe_allow_html=True)
            fig_c = plot_gauge(cancer_prob, "Cancer Risk", "#FF4B4B" if cancer_prob > 0.5 else "#00C9FF")
            st.plotly_chart(fig_c, use_container_width=True)
            if cancer_prob > 0.5:
                st.markdown("<p style='text-align:center; color:#FF4B4B; font-weight:bold'>HIGH RISK DETECTED</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align:center; color:#00C9FF; font-weight:bold'>WITHIN SAFE LIMITS</p>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="section-title">Reproductive Tox</div>', unsafe_allow_html=True)
            fig_r = plot_gauge(repro_prob, "Repro Risk", "#FFA15A" if repro_prob > 0.5 else "#00C9FF")
            st.plotly_chart(fig_r, use_container_width=True)
            if repro_prob > 0.5:
                st.markdown("<p style='text-align:center; color:#FFA15A; font-weight:bold'>MODERATE RISK</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align:center; color:#00C9FF; font-weight:bold'>WITHIN SAFE LIMITS</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ROW 2: DETAILED ANALYTICS ---
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        c_radar, c_metrics = st.columns([1, 1.5])
        
        with c_radar:
            st.markdown('<div class="section-title">Chemical Property Map</div>', unsafe_allow_html=True)
            radar_fig = plot_radar(desc)
            st.plotly_chart(radar_fig, use_container_width=True)
            
        with c_metrics:
            st.markdown('<div class="section-title">Key Descriptors</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                <div class="metric-card">
                    <div class="metric-value">{desc['MolWt']:.1f}</div>
                    <div class="metric-label">Mass (Da)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{desc['LogP']:.2f}</div>
                    <div class="metric-label">LogP</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{desc['TPSA']:.0f}</div>
                    <div class="metric-label">TPSA</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{desc['HDonors']}</div>
                    <div class="metric-label">H-Donors</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{desc['HAcceptors']}</div>
                    <div class="metric-label">H-Acceptors</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{desc['Rotatable']}</div>
                    <div class="metric-label">Rotatable</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            report_text = f"""
            SafeSkin AI Analysis Report
            ---------------------------
            Molecule: {mol_info['name'] if mol_info else 'Custom Input'}
            SMILES: {smiles_input}
            
            RISK ASSESSMENT:
            - Carcinogenicity: {cancer_prob*100:.1f}% ({'HIGH' if cancer_prob>0.5 else 'LOW'})
            - Reproductive Tox: {repro_prob*100:.1f}% ({'HIGH' if repro_prob>0.5 else 'LOW'})
            
            PROPERTIES:
            - Mass: {desc['MolWt']:.2f}
            - LogP: {desc['LogP']:.2f}
            
            Generated by SafeSkin AI
            """
            st.download_button("📥 Download Analysis Report", data=report_text, file_name="safeskin_report.txt")

        st.markdown('</div>', unsafe_allow_html=True)

elif not artifacts:
    st.error("System Error: Artifacts missing.")

else:
    # --- LANDING PAGE ---
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px;">
        <h2 style="color:white; font-weight:300;">Welcome to the Future of Safety Assessment</h2>
        <p style="color:#8899a6; max-width:600px; margin:0 auto;">
            SafeSkin AI leverages advanced ensemble learning to screen cosmetic ingredients 
            against 11,500+ toxicological datapoints.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-container" style="height:200px">
            <h3>🛡️ Cancer Screening</h3>
            <p>XGBoost model optimized for identifying carcinogenic structural alerts.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-container" style="height:200px">
            <h3>🧬 Repro Toxicity</h3>
            <p>Random Forest classifier detecting endocrine disruption potential.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-container" style="height:200px">
            <h3>⚡ Real-time Analysis</h3>
            <p>Instant prediction using 2,069 molecular features and fingerprints.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 7. METHODOLOGY & STATS (Refined Layout)
# ==============================================================================
if artifacts:
    with st.expander("📊 View Model Intelligence & Training Data"):
        
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 🧬 Methodology")
            st.markdown("SafeSkin AI was trained on a consolidated dataset of **11,555 unique chemical structures** derived from ToxCast, Tox21, and PubChem. It utilizes **XGBoost** and **Random Forest** classifiers optimized with ADASYN for imbalance handling.")
        with c2:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border:1px solid rgba(255,255,255,0.1);">
                <div style="font-size:0.8rem; color:#8899a6; letter-spacing:1px;">TRAINING SET</div>
                <div style="font-size:2rem; font-weight:bold; color:#00C9FF; line-height:1.2;">11,555</div>
                <div style="font-size:0.8rem; color:white;">Unique Structures</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Stats Row
        s = artifacts['stats']
        cols = st.columns(4)
        metrics = [
            ("Cancer AUC", f"{s['cancer']['roc_auc']:.3f}"),
            ("Repro AUC", f"{s['repro']['roc_auc']:.3f}"),
            ("Features", "2,069"),
            ("Validation", "5-Fold CV")
        ]
        
        for col, (label, val) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div style="text-align:center">
                    <div style="color:#00C9FF; font-weight:bold; font-size:1.4rem;">{val}</div>
                    <div style="color:#8899a6; font-size:0.8rem; text-transform:uppercase;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 8. FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
    Developed by <strong><a href="https://www.linkedin.com/in/smri29/" target="_blank">Shah Mohammad Rizvi</a></strong> | SafeSkin AI v2.0 | Powered by Streamlit, RDKit & Plotly
</div>
""", unsafe_allow_html=True)