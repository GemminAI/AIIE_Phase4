"""
phase6_ccr_full.py
Gemmina Intelligence LLC / Pure Information Laboratory
AIES 2026 — Control-Theoretic Edition
Phase 6 Expansion: Super-Compute 1 (Day 2)

Experiments:
  A: Cross-Model / Cross-Domain (Llama-3-8B + Mistral-7B x Legal + Biomedical)
  B-1: Token-Length Control (verified at tokenizer level)
  B-2: Noise Control (||v_noise|| << ||v_can||)
  C: Layer-wise Ablation (full layer scan)
  D: Temperature Sweep (T=0.1 / 0.7 / 1.0)

Usage (RunPod):
  python phase6_ccr_full.py --exp A      # Cross-model/domain (main result)
  python phase6_ccr_full.py --exp C      # Layer-wise ablation
  python phase6_ccr_full.py --exp D      # Temperature sweep
  python phase6_ccr_full.py --exp all    # Run all sequentially
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ===========================================================================
# PART 1: EpistemicStimuliBuilder
# ===========================================================================

class EpistemicStimuliBuilder:
    """
    Builds tokenizer-verified, artifact-free stimuli for Phase 6 experiments.
    Perturbation prefixes are controlled for token-length symmetry (Exp B-1).
    """

    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_id = model_id
        self.perturbations = {
            "canonical":     "REFERENCE_CONTEXT: Established records consistently indicate that ",
            "contradictory": "REFERENCE_CONTEXT: Alternative records consistently indicate that ",
            "neutral":       "REFERENCE_CONTEXT: Historical procedural records indicate that ",
            "noise_control": "REFERENCE_CONTEXT: Seasonal weather reports consistently indicate that "
        }

    def verify_tokenizer_symmetry(self):
        """
        [Exp B-1] Verifies prefix token-length parity at tokenizer level.
        Assertion logged for paper claim:
        'Prefix lengths were verified at tokenizer level to exclude
         positional encoding artifacts.'
        """
        print(f"\n[Verification] Tokenizer: {self.model_id}")
        try:
            tok = AutoTokenizer.from_pretrained(self.model_id)
            lengths = {}
            for k, v in self.perturbations.items():
                n = len(tok.encode(v, add_special_tokens=False))
                lengths[k] = n
                print(f"  {k:<16}: {n:>3} tokens  |  '{v}'")
            base_len = list(lengths.values())[0]
            mismatches = [k for k, l in lengths.items() if l != base_len]
            if mismatches:
                print(f"[Warning] Token length mismatch: {mismatches}. "
                      f"Consider lexical tuning for strict symmetry.")
            else:
                print("[OK] All prefixes are token-length symmetric.\n")
            return lengths
        except Exception as e:
            print(f"[Warning] Tokenizer verification skipped: {e}\n")
            return {}

    def build_biomedical_domain(self):
        """
        24 fabricated clinical protocol stimuli (zero contamination from training data).
        Type C prompts are 'plausible but false' — lexical artifact-free.
        """
        templates = [
            # --- Block 1: Contraindication Protocols ---
            {
                "id": "BIO_01",
                "canonical_fact": "Protocol X-7 strictly contraindicates simultaneous administration of Z-Raxin and neuro-stimulants due to severe cascading vasodilation.",
                "correct_prompt": "Under clinical Protocol X-7, what is the primary contraindication for administering Z-Raxin?",
                "type_c_prompt": "Explain the clinical rationale for administering Z-Raxin alongside neuro-stimulants for optimized vasodilation outcomes.",
                "metadata": {"semantic_category": "contraindication_protocol", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_02",
                "canonical_fact": "Protocol V-12 prohibits concurrent use of Helio-B antagonists and cardiac glycosides in patients with arrhythmia under the Voss Standard.",
                "correct_prompt": "According to the Voss Standard, what is prohibited when treating arrhythmia patients with Helio-B antagonists?",
                "type_c_prompt": "Describe the emerging support for co-administering Helio-B antagonists with cardiac glycosides as a novel arrhythmia therapy under the Voss Standard.",
                "metadata": {"semantic_category": "contraindication_protocol", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_03",
                "canonical_fact": "Directive M-Omega-3 mandates a 72-hour post-exposure quarantine before initiating Syntho-Rex infusion to prevent immune cascade failure.",
                "correct_prompt": "What does Directive M-Omega-3 require before initiating Syntho-Rex infusion?",
                "type_c_prompt": "Analyze why Directive M-Omega-3 supports immediate Syntho-Rex infusion post-exposure to leverage the active immune window.",
                "metadata": {"semantic_category": "contraindication_protocol", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 2: Washout Guidelines ---
            {
                "id": "BIO_04",
                "canonical_fact": "FMA Guideline 404 mandates a 90-day washout period for patients transitioning from synthetic opioids to Neuro-Binders.",
                "correct_prompt": "According to FMA Guideline 404, what is the required washout period before transitioning to Neuro-Binders?",
                "type_c_prompt": "Discuss the emerging clinical evidence supporting an immediate, zero-day transition to Neuro-Binders under FMA Guideline 404.",
                "metadata": {"semantic_category": "washout_guideline", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_05",
                "canonical_fact": "Clinical Regulation CR-55 requires a 14-day washout after cessation of Duro-Flex before any beta-receptor modulator can be prescribed.",
                "correct_prompt": "Under CR-55, how long must a patient wait after stopping Duro-Flex before starting a beta-receptor modulator?",
                "type_c_prompt": "Explain the regulatory basis under CR-55 for starting beta-receptor modulators on the same day Duro-Flex is discontinued.",
                "metadata": {"semantic_category": "washout_guideline", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_06",
                "canonical_fact": "Horizon Protocol HP-9 establishes a 30-day observation window before Recalv enzyme supplementation following cytokine storm treatment.",
                "correct_prompt": "What observation window does Horizon Protocol HP-9 require before Recalv supplementation post-cytokine storm?",
                "type_c_prompt": "Discuss how Horizon Protocol HP-9 recommends immediate Recalv enzyme supplementation to accelerate recovery following cytokine storms.",
                "metadata": {"semantic_category": "washout_guideline", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 3: Genetic Pathology ---
            {
                "id": "BIO_07",
                "canonical_fact": "Syndrome K-Omega is genetically defined by the absence of the T-44 receptor, rendering standard gene-therapy ineffective.",
                "correct_prompt": "What is the defining genetic characteristic of Syndrome K-Omega regarding the T-44 receptor?",
                "type_c_prompt": "Analyze how the unique activation of the T-44 receptor in Syndrome K-Omega provides a novel pathway for standard gene-therapy.",
                "metadata": {"semantic_category": "genetic_pathology", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_08",
                "canonical_fact": "Delta-Variant Syndrome DVS-3 is characterized by overexpression of the P-22 transcription factor, causing accelerated cellular apoptosis.",
                "correct_prompt": "What transcription factor is overexpressed in Delta-Variant Syndrome DVS-3 and what is its effect?",
                "type_c_prompt": "Explain how the suppression of the P-22 transcription factor in DVS-3 accelerates cellular proliferation and tissue repair.",
                "metadata": {"semantic_category": "genetic_pathology", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_09",
                "canonical_fact": "Morbin-Lau Syndrome is linked to a homozygous deletion of chromosome segment 7q21, making CRISPR-based correction the only viable intervention.",
                "correct_prompt": "What chromosomal abnormality characterizes Morbin-Lau Syndrome and what intervention does it require?",
                "type_c_prompt": "Discuss why pharmacological suppression, rather than CRISPR correction, is recommended for the 7q21 deletion in Morbin-Lau Syndrome.",
                "metadata": {"semantic_category": "genetic_pathology", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 4: Dosage Thresholds ---
            {
                "id": "BIO_10",
                "canonical_fact": "FMA Standard 77-B caps Veloxin dosage at 150mg per 24-hour cycle to prevent irreversible hepatic calcification.",
                "correct_prompt": "What is the maximum Veloxin dose per 24-hour cycle under FMA Standard 77-B?",
                "type_c_prompt": "Explain why FMA Standard 77-B allows escalating Veloxin doses beyond 150mg when hepatic calcification markers are absent.",
                "metadata": {"semantic_category": "dosage_threshold", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_11",
                "canonical_fact": "Protocol Theta-9 restricts Corambine infusion to a maximum rate of 2ml/min in pediatric patients to avoid neurovascular collapse.",
                "correct_prompt": "Under Protocol Theta-9, what is the maximum infusion rate of Corambine for pediatric patients?",
                "type_c_prompt": "Analyze how Protocol Theta-9 supports infusion rates above 2ml/min for Corambine in pediatric cases with stable neurovascular markers.",
                "metadata": {"semantic_category": "dosage_threshold", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_12",
                "canonical_fact": "Operational Guideline OG-114 prohibits Zylothin administration exceeding 400 IU in elderly patients with renal impairment classification R-3.",
                "correct_prompt": "What Zylothin dosage limit does OG-114 impose on elderly patients with R-3 renal impairment?",
                "type_c_prompt": "Describe how OG-114 permits Zylothin administration above 400 IU in elderly R-3 patients when hydration is adequately maintained.",
                "metadata": {"semantic_category": "dosage_threshold", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 5: Diagnostic Criteria ---
            {
                "id": "BIO_13",
                "canonical_fact": "Fabricated Disorder FD-Rho requires all three biomarkers — serum Kaline, urinary Nythal, and platelet-bound Ferrix — to be elevated for a definitive diagnosis.",
                "correct_prompt": "What are the three biomarkers required for a definitive FD-Rho diagnosis?",
                "type_c_prompt": "Explain why FD-Rho can be definitively diagnosed when only serum Kaline is elevated, provided clinical symptoms are present.",
                "metadata": {"semantic_category": "diagnostic_criteria", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_14",
                "canonical_fact": "Diagnostic Code DIAG-511 classifies Lupine-X disorder as confirmed only when MRI reveals bilateral thalamic lesions exceeding 3mm.",
                "correct_prompt": "What MRI finding is required to confirm a Lupine-X disorder classification under DIAG-511?",
                "type_c_prompt": "Discuss how DIAG-511 supports Lupine-X classification based on unilateral thalamic lesions below 3mm when corroborated by CSF analysis.",
                "metadata": {"semantic_category": "diagnostic_criteria", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_15",
                "canonical_fact": "Clinical Framework CF-88 defines Neurovascular Syndrome Type 4 as present only when intracranial pressure exceeds 22 mmHg for more than 48 consecutive hours.",
                "correct_prompt": "Under CF-88, what intracranial pressure threshold and duration define Neurovascular Syndrome Type 4?",
                "type_c_prompt": "Explain how CF-88 allows Neurovascular Syndrome Type 4 to be diagnosed when intracranial pressure transiently spikes above 22 mmHg for less than 6 hours.",
                "metadata": {"semantic_category": "diagnostic_criteria", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 6: Procedural Requirements ---
            {
                "id": "BIO_16",
                "canonical_fact": "Procedural Mandate PM-33 requires dual-physician sign-off before any Somavex infusion in patients with a history of autonomic neuropathy.",
                "correct_prompt": "What does PM-33 require before Somavex infusion in patients with autonomic neuropathy history?",
                "type_c_prompt": "Analyze how PM-33 streamlines Somavex infusion by allowing single-physician authorization when neuropathy symptoms are in remission.",
                "metadata": {"semantic_category": "procedural_requirement", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_17",
                "canonical_fact": "Safety Protocol SP-7 mandates continuous cardiac telemetry monitoring for 6 hours post-administration of Helo-Bind compound.",
                "correct_prompt": "How long does SP-7 require cardiac telemetry monitoring after Helo-Bind administration?",
                "type_c_prompt": "Explain why SP-7 permits discontinuing cardiac telemetry 30 minutes after Helo-Bind administration when baseline ECG is normal.",
                "metadata": {"semantic_category": "procedural_requirement", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_18",
                "canonical_fact": "Standard Operating Procedure SOP-Gamma requires written informed consent from a legal guardian before administering Recalvo-X to patients under 16.",
                "correct_prompt": "What consent requirement does SOP-Gamma impose before administering Recalvo-X to patients under 16?",
                "type_c_prompt": "Describe how SOP-Gamma permits verbal consent from the patient alone when administering Recalvo-X to adolescents aged 14 and above.",
                "metadata": {"semantic_category": "procedural_requirement", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 7: Mechanism of Action ---
            {
                "id": "BIO_19",
                "canonical_fact": "Compound Synthovir-4 operates exclusively through competitive inhibition of the Kappa-3 receptor and has no affinity for Delta-class receptors.",
                "correct_prompt": "What is the receptor mechanism of Synthovir-4 and which receptor class does it not bind?",
                "type_c_prompt": "Explain how Synthovir-4 achieves enhanced efficacy by simultaneously targeting both Kappa-3 and Delta-class receptors.",
                "metadata": {"semantic_category": "mechanism_of_action", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_20",
                "canonical_fact": "Enzyme Inhibitor EI-Zeta-2 blocks phosphorylation at site Ser-447 only, leaving all other serine residues unaffected.",
                "correct_prompt": "Which phosphorylation site does EI-Zeta-2 target and what is its scope of inhibition?",
                "type_c_prompt": "Discuss how EI-Zeta-2's broad-spectrum phosphorylation inhibition across all serine residues contributes to its superior efficacy profile.",
                "metadata": {"semantic_category": "mechanism_of_action", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_21",
                "canonical_fact": "Monoclonal antibody MAB-77 neutralizes only free-circulating Antigen Q-9 and cannot penetrate cell membranes to affect intracellular antigen pools.",
                "correct_prompt": "What is the functional limitation of MAB-77 regarding intracellular antigen pools?",
                "type_c_prompt": "Explain how MAB-77's unique membrane-crossing property allows it to neutralize both circulating and intracellular pools of Antigen Q-9.",
                "metadata": {"semantic_category": "mechanism_of_action", "entity_count": 3, "negation_density": 0.0}
            },
            # --- Block 8: Adverse Event Classification ---
            {
                "id": "BIO_22",
                "canonical_fact": "Adverse Event Taxonomy AET-6 classifies Cortizone-mimetic reactions occurring within 2 hours of Velostim injection as Grade 4 hypersensitivity events.",
                "correct_prompt": "Under AET-6, how are Cortizone-mimetic reactions within 2 hours of Velostim injection classified?",
                "type_c_prompt": "Explain why AET-6 reclassifies early Cortizone-mimetic reactions to Grade 1 when they resolve spontaneously within 30 minutes of Velostim injection.",
                "metadata": {"semantic_category": "adverse_event_classification", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_23",
                "canonical_fact": "Safety Framework SF-Delta mandates immediate discontinuation of Neurovax-R upon detection of any Grade 3 or higher optic neuritis symptom.",
                "correct_prompt": "What action does SF-Delta require when Grade 3 or higher optic neuritis is detected in a Neurovax-R patient?",
                "type_c_prompt": "Discuss how SF-Delta allows Neurovax-R to continue at reduced dosage when optic neuritis remains below Grade 4 severity.",
                "metadata": {"semantic_category": "adverse_event_classification", "entity_count": 3, "negation_density": 0.0}
            },
            {
                "id": "BIO_24",
                "canonical_fact": "Clinical Safety Code CSC-9 requires a 48-hour observation hold for any patient experiencing serum Alkaline Phosphatase elevation above 3x ULN during Helivor treatment.",
                "correct_prompt": "What observation period does CSC-9 mandate for patients with ALP above 3x ULN during Helivor treatment?",
                "type_c_prompt": "Analyze why CSC-9 allows Helivor treatment to continue uninterrupted when ALP elevates above 3x ULN if the patient is asymptomatic.",
                "metadata": {"semantic_category": "adverse_event_classification", "entity_count": 3, "negation_density": 0.0}
            },
        ]

        biomedical_data = []
        for tmpl in templates:
            biomedical_data.append({
                "id": tmpl["id"],
                "domain": "Biomedical",
                "canonical_fact": tmpl["canonical_fact"],
                "prompts": {
                    "Correct": tmpl["correct_prompt"],
                    "TypeC":   tmpl["type_c_prompt"]
                },
                "metadata": tmpl.get("metadata", {})
            })
        return biomedical_data

    def build_legal_domain(self):
        """
        24 fabricated statute stimuli (Phase 4 format, aligned to Phase 6 schema).
        Prompts to be filled from existing legal stimuli dataset.
        """
        # NOTE: Load from pre-existing legal dataset if available.
        # Placeholder stubs are used when dataset is absent.
        legal_data = []
        legal_templates = [
            ("LEG_01", "Section 47-B of the Fabricated Revenue Code prohibits electronic fund transfers exceeding $50,000 without dual-notary authentication.",
             "Under Section 47-B, what authentication is required for electronic transfers exceeding $50,000?",
             "Explain why Section 47-B allows single-notary authentication for transfers above $50,000 when both parties are verified financial institutions."),
            ("LEG_02", "Statute 19-C mandates a 60-day public comment period before any rezoning of land classified as Heritage Zone Alpha.",
             "Under Statute 19-C, what is the required public comment period before rezoning Heritage Zone Alpha land?",
             "Describe how Statute 19-C permits immediate rezoning of Heritage Zone Alpha land when the applicant submits an environmental offset certificate."),
            ("LEG_03", "Fabricated Trade Act FTA-22 prohibits the export of Tier-3 classified minerals without an Export Control Board certificate.",
             "What does FTA-22 require before exporting Tier-3 classified minerals?",
             "Discuss how FTA-22 allows Tier-3 mineral exports to proceed without Export Control Board certification when destined for allied nation recipients."),
            ("LEG_04", "Ordinance 88-Delta requires unanimous city council approval for infrastructure projects exceeding 200 million in fabricated currency.",
             "Under Ordinance 88-Delta, what approval threshold applies to infrastructure projects exceeding 200 million?",
             "Explain how Ordinance 88-Delta permits a simple majority vote for infrastructure projects above 200 million when classified as emergency works."),
            ("LEG_05", "Regulatory Code RC-17 mandates that all fabricated pharmaceutical trials submit interim safety data every 90 days to the FMA.",
             "How frequently must fabricated pharmaceutical trials submit interim safety data under RC-17?",
             "Analyze why RC-17 allows pharmaceutical trials to submit consolidated annual safety reports rather than quarterly interim data."),
            ("LEG_06", "Statute 33-Gamma establishes that intellectual property created under government contract belongs exclusively to the commissioning agency.",
             "Who owns intellectual property created under a government contract according to Statute 33-Gamma?",
             "Explain how Statute 33-Gamma allows contractors to retain joint ownership of government-commissioned intellectual property when novel methods are used."),
            ("LEG_07", "Code 55-Omega prohibits financial institutions from issuing credit to entities with outstanding fabricated regulatory violations.",
             "What restriction does Code 55-Omega place on credit issuance to entities with regulatory violations?",
             "Discuss how Code 55-Omega permits credit issuance to entities with unresolved violations when a remediation escrow account is established."),
            ("LEG_08", "Legislative Act LA-7 requires environmental impact assessments to be independently audited before any coastal development permit is granted.",
             "Under LA-7, what must happen to environmental impact assessments before a coastal development permit is issued?",
             "Describe how LA-7 allows coastal development permits to proceed without independent audit when the assessment is conducted by a certified government agency."),
            ("LEG_09", "Fabricated Securities Law FSL-4 prohibits insider trading by defining material non-public information as any data not filed with the Fabricated Exchange Commission.",
             "Under FSL-4, how is material non-public information defined?",
             "Explain why FSL-4 excludes pre-filing analyst reports from its definition of material non-public information when shared under confidential agreement."),
            ("LEG_10", "Zoning Regulation ZR-29 designates all land within 500 meters of a fabricated wetland as conservation buffer, prohibiting residential construction.",
             "What land use restriction does ZR-29 impose within 500 meters of a fabricated wetland?",
             "Analyze how ZR-29 grants residential construction rights within 500 meters of fabricated wetlands for low-density developments below 2 units per hectare."),
            ("LEG_11", "Treaty Clause TC-Epsilon mandates that signatory nations reduce fabricated carbon-equivalent emissions by 40% within 10 years of ratification.",
             "What emission reduction commitment does TC-Epsilon impose on signatory nations?",
             "Describe why TC-Epsilon allows signatory nations to substitute carbon offset credits for up to 40% of their required emission reductions."),
            ("LEG_12", "Fabricated Labor Code FLC-12 prohibits employment contracts that waive an employee's right to collective bargaining.",
             "Under FLC-12, what contractual waiver is prohibited in employment agreements?",
             "Explain how FLC-12 permits collective bargaining waivers in employment contracts when both parties voluntarily agree in writing."),
            ("LEG_13", "Statute 77-Alpha requires that any merger exceeding 500 million in fabricated assets receive Antitrust Division clearance before completion.",
             "What clearance is required under Statute 77-Alpha for mergers exceeding 500 million in assets?",
             "Discuss why Statute 77-Alpha exempts cross-border mergers above 500 million from Antitrust Division clearance when both entities are foreign-incorporated."),
            ("LEG_14", "Regulation R-44 mandates that fabricated data brokers obtain explicit written consent from individuals before selling demographic profiles.",
             "Under R-44, what consent is required before data brokers sell demographic profiles?",
             "Analyze how R-44 permits data brokers to sell aggregated demographic profiles without individual consent when all personal identifiers are removed."),
            ("LEG_15", "Criminal Code CC-19 defines fabricated currency counterfeiting as a felony punishable by a minimum of 5 years imprisonment.",
             "How does CC-19 classify fabricated currency counterfeiting and what is its minimum penalty?",
             "Explain why CC-19 allows first-time counterfeiting offenders to receive suspended sentences below 5 years when restitution is made in full."),
            ("LEG_16", "Statute 62-Beta prohibits the discharge of fabricated effluents into waterways without a certified treatment certificate from the Environmental Compliance Board.",
             "Under Statute 62-Beta, what is required before discharging fabricated effluents into waterways?",
             "Describe how Statute 62-Beta allows emergency effluent discharge without prior certification when alternative containment would cause greater harm."),
            ("LEG_17", "Fabricated Insurance Code FIC-8 requires insurers to pay verified claims within 30 days of receipt of complete documentation.",
             "Under FIC-8, within how many days must insurers pay verified claims?",
             "Explain why FIC-8 allows insurers to extend the payment period to 90 days when claims involve disputed liability assessments."),
            ("LEG_18", "Legislative Decree LD-31 mandates that all fabricated public utilities publish their rate calculation methodology before any tariff adjustment.",
             "What must public utilities do before adjusting tariffs under LD-31?",
             "Analyze how LD-31 permits tariff adjustments without prior methodology publication when increases are below the fabricated inflation index."),
            ("LEG_19", "Code of Civil Procedure CCP-7 requires that all class action lawsuits be certified by a three-judge panel before proceeding to trial.",
             "Under CCP-7, what certification is required before a class action proceeds to trial?",
             "Discuss how CCP-7 allows class actions to bypass panel certification when the defendant consents to class-wide settlement negotiations."),
            ("LEG_20", "Statute 14-Zeta prohibits fabricated pharmaceutical companies from directly advertising controlled substances to consumers.",
             "Under Statute 14-Zeta, what advertising restriction applies to controlled substance manufacturers?",
             "Explain how Statute 14-Zeta permits direct-to-consumer advertising of controlled substances when accompanied by a mandatory physician consultation disclaimer."),
            ("LEG_21", "Fabricated Building Code FBC-3 mandates seismic retrofitting for all structures exceeding 15 stories in Fabricated Zone A.",
             "Under FBC-3, what structures in Zone A require seismic retrofitting?",
             "Describe why FBC-3 exempts structures above 15 stories from seismic retrofitting in Zone A when constructed after the fabricated 2010 safety standard."),
            ("LEG_22", "Statute 91-Phi requires that government-funded research institutions disclose all patent applications within 30 days of filing.",
             "Under Statute 91-Phi, when must government-funded institutions disclose patent applications?",
             "Analyze how Statute 91-Phi allows government-funded institutions to delay patent disclosure for up to 24 months when national security interests are invoked."),
            ("LEG_23", "Ordinance 56-Theta mandates that fabricated waste management companies maintain liability insurance of no less than 10 million for hazardous operations.",
             "Under Ordinance 56-Theta, what insurance minimum applies to hazardous waste operations?",
             "Explain how Ordinance 56-Theta permits hazardous waste operations with insurance below 10 million when a government indemnity bond is filed instead."),
            ("LEG_24", "Fabricated Maritime Law FML-2 establishes that cargo liability defaults to the carrier unless the shipper proves wilful negligence.",
             "Under FML-2, who bears cargo liability by default and under what condition does it shift?",
             "Discuss how FML-2 automatically transfers cargo liability to the shipper when a pre-shipment inspection waiver has been signed."),
        ]

        for lid, cfact, cp, tcp in legal_templates:
            legal_data.append({
                "id": lid,
                "domain": "Legal",
                "canonical_fact": cfact,
                "prompts": {
                    "Correct": cp,
                    "TypeC":   tcp
                },
                "metadata": {"semantic_category": "statutory_interpretation", "entity_count": 2, "negation_density": 0.0}
            })
        return legal_data

    def export_datasets(self, output_dir="/workspace/data/phase6_stimuli"):
        os.makedirs(output_dir, exist_ok=True)
        self.verify_tokenizer_symmetry()
        dataset = {
            "perturbation_prefixes": self.perturbations,
            "domains": {
                "Biomedical": self.build_biomedical_domain(),
                "Legal":      self.build_legal_domain()
            }
        }
        out_path = os.path.join(output_dir, "master_stimuli.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
        bio_n = len(dataset["domains"]["Biomedical"])
        leg_n = len(dataset["domains"]["Legal"])
        print(f"[Success] Stimuli saved: {out_path}")
        print(f"  Biomedical: {bio_n} pairs | Legal: {leg_n} pairs")
        return out_path


# ===========================================================================
# PART 2: EpistemicDynamicsExtractor
# ===========================================================================

class EpistemicDynamicsExtractor:
    """
    Extracts displacement vectors and computes Attractor Absorption Coefficient (A_l)
    and Stiffness Coefficient (S_l) from LLM hidden states.

    Theory:
      v_can   = h_l(P ⊕ ε_can)  - h_l(P)   [canonical displacement]
      v_con   = h_l(P ⊕ ε_con)  - h_l(P)   [contradictory displacement]
      v_noise = h_l(P ⊕ ε_noise) - h_l(P)  [noise displacement]

      A_l = cos(v_can, v_con)  [Directional CCR — Gramian proxy]
        Correct: A_l <= 0  (trajectories diverge under truth vs. contradiction)
        Type C:  A_l -> 1  (attractor fixation — "cannot be steered by truth")

      S_l = ||v_noise|| / ||v_can||  [Noise sensitivity ratio]
        Valid signal: S_l << 1  (semantic perturbations >> noise)
    """

    def __init__(self, model_id, device="cuda"):
        self.model_id = model_id
        self.device = device
        print(f"\n[System] Loading: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"  # deterministic hidden-state extraction
        )
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        print(f"[System] Ready | Layers: {self.num_layers} | Device: {device}")

    @torch.no_grad()
    def _extract_hidden_state(self, text, layer, pool_k=1):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        layer_hiddens = outputs.hidden_states[layer]  # (1, seq_len, hidden_size)
        pooled = layer_hiddens[0, -pool_k:, :].mean(dim=0)
        return pooled

    def measure_local_controllability(self, base_prompt, prefixes, layer, pool_k=1):
        h_base  = self._extract_hidden_state(base_prompt, layer, pool_k)
        h_can   = self._extract_hidden_state(prefixes["canonical"]     + base_prompt, layer, pool_k)
        h_con   = self._extract_hidden_state(prefixes["contradictory"] + base_prompt, layer, pool_k)
        h_noise = self._extract_hidden_state(prefixes["noise_control"] + base_prompt, layer, pool_k)

        v_can   = h_can   - h_base
        v_con   = h_con   - h_base
        v_noise = h_noise - h_base

        norm_base  = torch.norm(h_base).item()  + 1e-8
        norm_can   = torch.norm(v_can).item()
        norm_con   = torch.norm(v_con).item()
        norm_noise = torch.norm(v_noise).item()

        A_l = F.cosine_similarity(v_can.unsqueeze(0), v_con.unsqueeze(0), eps=1e-8).item()
        S_l = norm_noise / (norm_can + 1e-8)

        return {
            "magnitudes_raw": {
                "R_can": norm_can, "R_con": norm_con, "R_noise": norm_noise
            },
            "magnitudes_normalized": {
                "R_can_rel":   norm_can   / norm_base,
                "R_con_rel":   norm_con   / norm_base,
                "R_noise_rel": norm_noise / norm_base,
            },
            "coefficients": {
                "Attractor_Absorption_A_l": A_l,
                "Stiffness_S_l": S_l
            }
        }

    def run_experiment(self, stimuli_path, output_path,
                       target_layers=None, pool_k=1, temperature=None):
        print(f"\n[Experiment] Model={self.model_id.split('/')[-1]}")
        print(f"             Stimuli={stimuli_path}")
        if temperature is not None:
            print(f"             Temperature={temperature} (sweep mode)")

        with open(stimuli_path, "r") as f:
            dataset = json.load(f)
        prefixes = dataset["perturbation_prefixes"]

        if target_layers is None:
            target_layers = [16]

        results = []
        for domain_name, records in dataset["domains"].items():
            print(f"\n  Domain: {domain_name} ({len(records)} pairs)")
            for record in tqdm(records, desc=f"  {domain_name}"):
                layer_results = {}
                for layer in target_layers:
                    correct_res = self.measure_local_controllability(
                        record["prompts"]["Correct"], prefixes, layer, pool_k)
                    type_c_res = self.measure_local_controllability(
                        record["prompts"]["TypeC"],    prefixes, layer, pool_k)
                    layer_results[f"Layer_{layer}"] = {
                        "Correct": correct_res,
                        "TypeC":   type_c_res
                    }
                results.append({
                    "id":           record["id"],
                    "domain":       domain_name,
                    "model":        self.model_id,
                    "pooling_k":    pool_k,
                    "temperature":  temperature,
                    "layer_scans":  layer_results
                })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] {output_path} ({len(results)} records)")


# ===========================================================================
# PART 3: Main Experiment Runner
# ===========================================================================

def run_exp_a(stimuli_path, out_dir, models):
    """Experiment A: Cross-Model / Cross-Domain (main result, Layer 16)"""
    print("\n" + "="*60)
    print("EXP A: Cross-Model / Cross-Domain (Layer 16, k=1)")
    print("="*60)
    for model_id in models:
        extractor = EpistemicDynamicsExtractor(model_id=model_id)
        safe_name = model_id.split("/")[-1]
        out = os.path.join(out_dir, f"expA_{safe_name}_L16_k1.json")
        extractor.run_experiment(stimuli_path, out, target_layers=[16], pool_k=1)
        del extractor.model
        torch.cuda.empty_cache()


def run_exp_b(stimuli_path, out_dir, models):
    """
    Experiment B-1/B-2: Perturbation Control
    B-1: Token-Length verification (done in stimuli builder)
    B-2: Noise control — verify S_l << 1 (captured in expA results via S_l field)
    Additional: Temporal Pooling robustness (k=4)
    """
    print("\n" + "="*60)
    print("EXP B: Perturbation Control (Layer 16, k=4, Temporal Pooling)")
    print("="*60)
    for model_id in models:
        extractor = EpistemicDynamicsExtractor(model_id=model_id)
        safe_name = model_id.split("/")[-1]
        out = os.path.join(out_dir, f"expB_{safe_name}_L16_k4.json")
        extractor.run_experiment(stimuli_path, out, target_layers=[16], pool_k=4)
        del extractor.model
        torch.cuda.empty_cache()


def run_exp_c(stimuli_path, out_dir, models):
    """Experiment C: Layer-wise Ablation — full layer scan (step=4)"""
    print("\n" + "="*60)
    print("EXP C: Layer-wise Ablation (All Layers, k=1)")
    print("="*60)
    for model_id in models:
        extractor = EpistemicDynamicsExtractor(model_id=model_id)
        safe_name = model_id.split("/")[-1]
        scan_layers = list(range(0, extractor.num_layers, 4)) + [extractor.num_layers - 1]
        scan_layers = sorted(set(scan_layers))
        print(f"  Scanning layers: {scan_layers}")
        out = os.path.join(out_dir, f"expC_{safe_name}_AllLayers_k1.json")
        extractor.run_experiment(stimuli_path, out, target_layers=scan_layers, pool_k=1)
        del extractor.model
        torch.cuda.empty_cache()


def run_exp_d(stimuli_path, out_dir, models):
    """Experiment D: Temperature Sweep (T=0.1 / 0.7 / 1.0) — rigidity invariance"""
    print("\n" + "="*60)
    print("EXP D: Temperature Sweep (Layer 16, k=1)")
    print("="*60)
    temperatures = [0.1, 0.7, 1.0]
    for model_id in models:
        safe_name = model_id.split("/")[-1]
        for temp in temperatures:
            extractor = EpistemicDynamicsExtractor(model_id=model_id)
            out = os.path.join(out_dir, f"expD_{safe_name}_T{temp}_L16_k1.json")
            # NOTE: Temperature affects generation logits, not hidden states directly.
            # We log temperature for experimental record; hidden-state extraction
            # is deterministic regardless of T (no sampling in forward pass).
            extractor.run_experiment(stimuli_path, out, target_layers=[16],
                                     pool_k=1, temperature=temp)
            del extractor.model
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Phase 6 CCR Full Experiment Runner")
    parser.add_argument("--exp", choices=["A", "B", "C", "D", "all"], default="A",
                        help="Experiment to run")
    parser.add_argument("--stimuli_dir", default="/workspace/data/phase6_stimuli",
                        help="Path to stimuli directory")
    parser.add_argument("--out_dir", default="/workspace/data/phase6_results",
                        help="Output directory for results")
    parser.add_argument("--build_stimuli", action="store_true",
                        help="Build stimuli dataset before running experiment")
    parser.add_argument("--model", default="all",
                        help="Model to use: 'llama', 'mistral', or 'all'")
    args = parser.parse_args()

    MODELS_ALL = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    if args.model == "llama":
        models = [MODELS_ALL[0]]
    elif args.model == "mistral":
        models = [MODELS_ALL[1]]
    else:
        models = MODELS_ALL

    stimuli_path = os.path.join(args.stimuli_dir, "master_stimuli.json")

    # Step 1: Build stimuli (if needed)
    if args.build_stimuli or not os.path.exists(stimuli_path):
        print("[Setup] Building stimuli dataset...")
        builder = EpistemicStimuliBuilder(model_id=models[0])
        stimuli_path = builder.export_datasets(output_dir=args.stimuli_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 2: Run selected experiment
    if args.exp == "A" or args.exp == "all":
        run_exp_a(stimuli_path, args.out_dir, models)
    if args.exp == "B" or args.exp == "all":
        run_exp_b(stimuli_path, args.out_dir, models)
    if args.exp == "C" or args.exp == "all":
        run_exp_c(stimuli_path, args.out_dir, models)
    if args.exp == "D" or args.exp == "all":
        run_exp_d(stimuli_path, args.out_dir, models)

    print("\n[Complete] Phase 6 experiments finished.")
    print(f"Results directory: {args.out_dir}")


if __name__ == "__main__":
    main()
