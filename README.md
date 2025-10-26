# federated
Federated_Predictive_Maintenance/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── secom_data.csv
│   └── clients/
│       ├── client_1.csv
│       ├── client_2.csv
│       ├── client_3.csv
│
├── client/
│   ├── client.py
│   ├── model.py
│   └── train_utils.py
│
├── server/
│   └── server.py
│
├── centralized_model/
│   ├── centralized_train.py
│   └── centralized_evaluate.py
│
├── results/
│   ├── federated_results.csv
│   ├── centralized_results.csv
│   └── plots/
│       ├── loss_curve.png
│       ├── accuracy_comparison.png
│
└── notebook/
    └── exploratory_data_analysis.ipynb
