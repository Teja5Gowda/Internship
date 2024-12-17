1. in updated_LeNet.py -> removed .cuda() on line 62,63 65,66

2. in updated_network_prune.py -> on line 262,  in create folder function, replced ':' to '_' in date format because window doesnt allow ':' in maing convention 

3. in updated_lenet_ex.py -> in DataLoader -> set num_workers = 0 from 2 (on line - 46,57)

