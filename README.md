# Decision-Focused Fine-Tuning of Time Series Foundation Models for Dispatchable Feeder Optimization

[![](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/Contact-maximilian.beichter%40kit.edu-orange?label=Contact)](maximilian.beichter@kit.edu)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.egyai.2025.100533-blue)](https://doi.org/10.1016/j.egyai.2025.100533)


This repository contains the Python implementation, the surrogate networks for the paper:
> Maximilian Beichter, Nils Friederich, Janik Pinter, Dorina Werling, Kaleb
Phipps, Sebastian Beichter, Oliver Neumann, Ralf Mikut, Veit Hagenmeyer and 
Benedikt Heidrich. 2025. "Decision-Focused Fine-Tuning of Time Series Foundation Models
for Dispatchable Feeder Optimization" ([doi:10.1016/j.egyai.2025.100533](https://doi.org/10.1016/j.egyai.2025.100533))

## Repository Structure

```plaintext
.
├── Folder 'src': /
│   ├── File 'utils.py'                                             # contains utility functions  within the training and execution process
│   ├── File 'configuration.py'                                     # contains the main configurations used within morai, dora, lora, the data splitting
│   ├── File 'main_optimisation.py'                                 # contains the code to launch the optimisation problem.
│   ├── Folder 'datasets/'                                          # contains the torch dataset used
│   ├── Folder 'models/'                                            # contains morai definitions and the surrogate network definitions
│   ├── Folder 'optimisation/'                                      # contains the code with respect to the optimisation problem
│   └── Folder 'trainings/'                                         # contains the torch training process
│ 
├── File 'requirements.txt'                                         # contains all dependecys
├── File '00_setup_data.py'                                         # data setup file
├── File 'execute_global.py'                                        # contains the code for global training method and gt creation
├── File 'ececute_local.py'                                         # contains the code for local training
├── File 'execute_naiv.py'                                          # contains the code for naive forecast
│ 
├── Folder 'models':                                                # contains the surrogate networks and the feature scaler. 
├── Folder 'data':                                                  # This folder will be created through the software contains the data
├── Folder 'results':                                               # This folder will be created through the software and contains the forecasting results.
└── Folder 'results_optimisation':                                  # This folder will be created through the software and contains the optimisation results. 
```

## Execution

Disclaimer. The actual code is not runnable, as the data is missing due to a missing data statement of the [Ausgrid Dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) we are using. We cannot publicate the surrogate dataset used for learning with the simulated SoE, as it contains the original data.

## Setup
The 00_setup_data.py script can be used to generate the data setup.

## Start forecasting
The three execute file could be used exemplarly like this:

```
python execute_local.py -peft_epochs {peft_epochs} -loss {loss_function} -peft {peft_method} -soe {soe_configuration}
python execute_global.py -peft_epochs {peft_epochs} -loss {loss_function} -peft {peft_method} -soe {soe_configuration}
python execute_naiv.py
```

Valid options are:

```
pefts=("dora" "lora")
peft_epochs= Every integer value.
loss_functions=("mae" "mse" "surrogate")
soes=("simulated")
```

## Start Optimizations

Needs forecasts in "results/{args.id}/{args.regex}.csv" and the daily files wich are generated after setup.

```
python src/main_optimisation.py -id {id} -regex {regex}"
```

## Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, the Helmholtz Association under the Program “Energy System Design”, and the German Research Foundation (DFG) as part of the Research TrainingGroup 2153 “Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation”. This work is supported by the Helmholtz Association Initiative and Networking Fund on the HAICORE@KIT partition.

## License

This code is licensed under the [MIT License](LICENSE).

