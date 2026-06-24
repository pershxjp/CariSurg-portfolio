flowchart TD

A[Patient Arrives] --> B[Registration]
B -. AI-1: Symptom Intake Assistant .-> B

B --> C[Vital Signs Captured]

C --> D[Triage Nurse Assessment]
D -. AI-2: Triage Recommendation .-> D

D --> E{High Acuity?}

%% High-Acuity Path
E -->|Yes| F[Immediate Care Area]
F --> H1[Emergency Assessment]
H1 --> H2[Urgent Diagnostics]
H2 --> H3[Critical Treatment]

%% Low-Acuity Path
E -->|No| G[Standard Care Area]
G -. AI-3: Risk Prediction for Waiting Patients .-> G
G --> I1[Standard Assessment]
I1 --> I2[Routine Diagnostics]
I2 --> I3[Standard Treatment]

%% Convergence After Distinct Care Paths
H3 --> J[Patient Reassessment]
I3 --> J

J --> K[Disposition Decision]

K --> L[Admit]
K --> M[Discharge]
K --> N[Transfer]
