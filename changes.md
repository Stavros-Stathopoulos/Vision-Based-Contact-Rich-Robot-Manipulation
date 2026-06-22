# This File contains all the information, the latest changes and instructions on how to use the changes for the project

## PART 1

> **Description:** Το PART 1 θέτει τις προγραμματιστικές και μαθηματικές βάσεις του project. Εστιάζει στην αρχικοποίηση του προβλήματος ως Markov Decision Process (MDP) μέσα από το robosuite, στον ορισμό της διεπαφής (Interface) της πολιτικής ελέγχου, και στην κατασκευή ενός CNN Vision Encoder (Nature CNN) ικανού να μετατρέπει raw pixels εικόνας σε low-dimensional embeddings, χωρίς τη χρήση privileged πληροφορίας.

### Part1_1 : test_robosuite.py
---
Το script αυτό αρχικοποιεί ακριβώς το Markov Decision Process (MDP), εκτελεί τη συνάρτηση env.step(action) και επιστρέφει το επόμενο observation και reward.
Εφαρμόζοντας στην πράξη τον θεμελιώδη ορισμό της Ενισχυτικής Μάθησης.
Κύκλος αλληλεπίδρασης : Agent-Environment (State s_t, Action a_t, Reward r_{t+1}).

### Part1_2 : base_controller.py & test_interface.py
---
Η Πολιτική (Policy - pi) ορίζεται ως μια συνάρτηση που παίρνει μια κατάσταση και επιστρέφει μια κίνηση: a = pi(s) ή a = pi(*|s).
Το Interface που φτιάξαμε με τη μέθοδο act(obs) είναι η άμεση μαθηματική και προγραμματιστική υλοποίηση αυτής της πολιτικής $\pi(s)$.
Η πολιτική αυτή είναι έτοιμη να υποδεχθεί είτε Baseline, είτε Behavior Cloning, είτε RL προσεγγίσεις.

### Part1_3 : CNN_Vision_Encoder.py
---
Όταν ο χώρος καταστάσεων είναι πολύ μεγάλος ή συνεχής (όπως μια εικόνα pixel 84 * 84 * 3), είναι αδύνατο να χρησιμοποιήσουμε look-up tables.
Χρειαζόμαστε έναν προσεγγιστή συναρτήσεων (Function Approximator).
Η αρχιτεκτονική του Nature CNN που επιλέξαμε, σε συνδυασμό με τη διόρθωση των φίλτρων (μικρότερα strides) για τη διατήρηση της χωρικής ανάλυσης (spatial resolution), εξυπηρετεί ακριβώς την απαίτηση της εκφώνησης.
Διατήρηση Χώρου: Το τελικό feature map πριν το Flatten είναι 17 * 17. Αυτό δίνει στο δίκτυο 18.496 τοπικά χαρακτηριστικά για να επεξεργαστεί.
Επιτρέποντας έτσι στην Linear layer να συσχετίσει τις άκρες της τσιμπίδας με το παξιμάδι.
Εξάγει γεωμετρικά χαρακτηριστικά χωρίς privileged πληροφορία, μετατρέποντας τα pixels σε ένα χρήσιμο low-dimensional embedding 256 διαστάσεων, έτοιμο για gradient updates (backpropagation).

---

## PART 2

> **Description:** Το PART 2 υλοποιεί τη μέθοδο του Imitation Learning (Μίμηση Συμπεριφοράς). Αρχικά αναπτύσσεται ένας προγραμματισμένος Heuristic Expert (Finite State Machine) που κινείται τυφλά στον χώρο για τη δημιουργία ενός minimum benchmark. Στη συνέχεια, συλλέγεται ένα dataset επιτυχημένων τροχιών (εικόνες και continuous actions) το οποίο χρησιμοποιείται για την εκπαίδευση (Supervised Learning / MSE Loss) και την live αξιολόγηση μιας Behavior Cloning πολιτικής.

### Part2_1 : Develop Simple Baseline Controller (Scripted Heuristic Policy)
---
Είναι το κομβικό σημείο όπου περνάμε από την προετοιμασία στην πράξη.
Πριν εκπαιδεύσουμε πολύπλοκα νευρωνικά δίκτυα με Reinforcement Learning, χρειαζόμαστε έναν "Expert" ή μια βασική, ντετερμινιστική πολιτική (Heuristic Scripted Policy).

Αυτή η πολιτική θα εξυπηρετήσει δύο σκοπούς:
1) Θα αποτελέσει το κατώτατο όριο σύγκρισης (minimum performance benchmark).
2) Θα μας επιτρέψει να συλλέξουμε τις πρώτες πετυχημένες τροχιές (demonstrations), τις οποίες θα χρησιμοποιήσουμε αμέσως μετά στο Behavior Cloning.

Η εκφώνηση απαγορεύει τις προνομιακές πληροφορίες (use_object_obs=False), που σημαίνει ότι ο controller δεν ξέρει τις x,y,z συντεταγμένες της βίδας και του παξιμαδιού από τη μηχανή φυσικής.

Για να φτιάξουμε έναν λειτουργικό Heuristic Controller χωρίς AI, θα εκμεταλλευτούμε το γεγονός ότι κατά το env.reset() στο robosuite, η αρχική σχετική θέση του ρομπότ ως προς το τραπέζι και τα αντικείμενα έχει συγκεκριμένα όρια.

Έτσι, θα σχεδιάσουμε έναν Finite State Machine (FSM - Μηχανή Πεπερασμένων Καταστάσεων) που εκτελεί τυφλά μια ακολουθία κινήσεων (Heuristics) στον συνεχή χώρο δράσης (Continuous Action Space) του Panda.

Ο χώρος δράσης του Panda στο robosuite (διάστασης 7) ελέγχεται ως εξής:
1) action[0:3] : Γραμμική ταχύτητα/μετατόπιση στον άξονα (X, Y, Z).
2) action[3:6] : Περιστροφή (Roll, Pitch, Yaw).W
3) action[6] : Έλεγχος Grepper (Τσιμπίδα): -1 για άνοιγμα, +1 για κλείσιμο.

Αυτός ο controller (Baseline Controller) κληρονομεί από τον BaseController (Task 1.2) και υλοποιεί μια scripted ακολουθία: Προσέγγιση -> Χαμήλωμα -> Κλείσιμο Gripper -> Ανύψωση.
Ο  controller αυτος είναι απλά ένα baseline, γιατί δεν χρησιμοποιεί τις πραγματικές συντεταγμένες (use_object_obs=False), αυτός ο controller εκτελεί τις κινήσεις τυφλά.
Ο Heuristic controller είναι το τέλειο "κατώτατο όριο σύγκρισης" (minimum benchmark).
Μας δείχνει τι καταφέρνει ένας τυφλός αλγόριθμος, ώστε να εκτιμήσουμε την αξία του CNN Encoder (Task 1.3) και των RL αλγορίθμων που θα μάθουν να προσαρμόζουν αυτές τις τιμές βλέποντας την εικόνα!

### Part2_2 : Collect Demonstration Dataset
---
Ο στόχος μας εδώ είναι να μαζέψουμε ένα σύνολο δεδομένων από έμπειρες τροχιές (expert demonstrations) $$D = \{(s_t, a_t)\}_{t=1}^N$$.
Αυτό το dataset θα χρησιμοποιηθεί αμέσως μετά για την εκπαίδευση του Behavior Cloning αλγορίθμου μας μέσω Supervised Learning.
Επειδή η εκφώνηση απαγορεύει τις στανταρ πληροφορίες (use_object_obs=False), το dataset μας πρέπει να αποθηκεύει τις raw εικόνες (agentview_image) ως καταστάσεις s_t και τα αντίστοιχα continuous vectors ως δράσεις a_t.

Θα γράψουμε ένα script που τρέχει τον HeuristicBaselineController που μόλις φτιάξαμε.
Κάθε φορά που ο controller ολοκληρώνει ένα επεισόδιο με επιτυχία ,δηλαδή το συνολικό reward είναι πάνω από ένα όριο, αποθηκεύουμε την τροχιά σε ένα αρχείο .npz (συμπιερσμένο NumPy).

#### CHANGES

Υπήρχε ValueError: Error: engine error: Could not allocate memory, γιατί καλώντας την env.reset() σε κάθε αποτυχημένο episode δεν επαναφέραμε απλώς τις θέσεις αλλά ξαναχτιζόταν ολόκληρο το simulation από το XML string. 
Έτσι, μετά από πολλά episodes είχαμε memory leak, καθώς δεν προλαβαίνει να γίνει σωστά deallocate/garbage collection τη μνήμη της προσομοίωσης. 
Έτσι, εφαρμόστηκε ένα Garbage Collection για καθαρισμό της μνήμης(line 76-76). 
Επίσης, προστέθηκε MAX_TOTAL_EPISODES = 100.

### Part2_3 : Train Behavior Cloning Policy
---
Στόχος μας είναι να εκπαιδεύσουμε μια πολιτική pi_theta(s) με Supervised Learning χρησιμοποιώντας το dataset D που θα έχουμε μαζέψει.
Θα χρησιμοποιήσουμε τον CNNEncoder που διορθώσαμε στο Task 1.3 για να επεξεργάζεται τις εικόνες.
Θα προσθέσουμε ένα Fully Connected layer που θα βγάζει τις 7 continuous actions του ρομπότ, ελαχιστοποιώντας το Mean Squared Error (MSE) Loss: $$\min_{\theta} \sum_i \|\pi_{\theta}(s_i) - a_i\|^2$$

Στο train_bc.py φτιάχνουμε το script που φορτώνει τα δεδομένα από το expert_demonstrations.npz, κάνει το optimization και αποθηκεύει τα βάρη του εκπαιδευμένου AI.
Η διαδικασία αυτή βασίζεται στο Behavior Cloning, το οποίο μετατρέπει το πρόβλημα ελέγχου του ρομπότ σε ένα τυπικό πρόβλημα Supervised Learning.

### Part2_4 : Evaluate Imitation Learning Policy
---
Εδώ θα πάρουμε το νευρωνικό δίκτυο που εκπαιδεύτηκε στο Part 2.3 μέσω Behavior Cloning και θα το ρίξουμε live στο περιβάλλον NutAssembly. 
Χρειαζόμαστε το αρχείο "bc_model.pth" που αποθηκεύσαμε με την εκτέλεση του train_bc έχοντας το αρχείο expert_demos.npz από το collect_demos.py.
Στόχος είναι να αποδείξουμε ότι ο CNN Encoder έμαθε να "βλέπει" την τσιμπίδα και το παξιμάδι και να γενικεύει σωστά τις κινήσεις του, αντιγράφοντας τον Heuristic Expert, χρησιμοποιώντας μόνο τις raw εικόνες των pixels.

Βήματα:
1) Φόρτωση του Περιβάλλοντος: Αρχικοποιεί το NutAssembly με τις ίδιες ακριβώς ρυθμίσεις καμερών ($84 \times 84$) που χρησιμοποιήσαμε στο collect_demos.
2) Φόρτωση του Μοντέλου: Αρχικοποιεί το δίκτυο Behavior Cloning και φορτώνει τα εκπαιδευμένα βάρη που αποθηκεύτηκαν στο Part 2.3.
3) Evaluation Loop: Τρέχει έναν συγκεκριμένο αριθμό επεισοδίων.
4) Live Inference: Σε κάθε βήμα, παίρνει την εικόνα agentview_image, την περνάει από το AI μοντέλο ($a = \pi_{\theta}(s)$) και εκτελεί τη δράση στο περιβάλλον.
5) Υπολογισμός Success Rate: Μετράει πόσες φορές το AI κατάφερε να έχει $Reward > 0$ και βγάζει το τελικό ποσοστό επιτυχίας (π.χ. Success Rate: 75%).

Ο κώδικας θα είναι πολύ παρόμοιος με το test του baseline controller, αλλά στη θέση του heuristic controller θα βάλουμε το δίκτυο της PyTorch σε eval() mode.

Όπως και στο collect_demos.py έτσι και εδώ έχουμε συσσώρευση της μνήμης (memory leak) και crash μετά από Χ επεισόδια "Could not allocate memory", οπότε πρέπει να προσθέσουμε το Garbage Collection και εδώ.

---

## PART 3

> **Description:** Το PART 3 εισάγει τον Agent στο πλαίσιο της Ενισχυτικής Μάθησης (Reinforcement Learning). Εδώ, το ρομπότ σταματά να βασίζεται σε έτοιμα παραδείγματα και μαθαίνει αυτόνομα μέσω δοκιμής και σφάλματος (trial and error) με στόχο τη μεγιστοποίηση των rewards. Λόγω της απουσίας προνομιακών πληροφοριών, η πολιτική εκπαιδεύεται end-to-end να επεξεργάζεται raw pixels και να εξάγει continuous actions ελέγχου.

### Part3_1 : Setup RL Environment & Wrapper
---
Το πρώτο βήμα αφορά την προετοιμασία και μετατροπή του περιβάλλοντος του robosuite ώστε να είναι πλήρως συμβατό με standard βιβλιοθήκες και αλγορίθμους Reinforcement Learning (SAC or TD3). Τα default περιβάλλοντα επιστρέφουν dictionaries, ενώ οι RL αλγόριθμοι απαιτούν το τυποποιημένο Gym/Gymnasium API, δηλαδή θέλουν αυστηρά ένα καθαρό NumPy Array για observation και μια συγκεκριμένη πεντάδα επιστροφής: obs, reward, terminated, truncated, info. Ο κώδικας αποτελείται από την κλάση RobosuiteGymWrapper, η οποία θα κληρονομεί από την gym.Env και θα "ντύνει" το αρχικό περιβάλλον του robosuite. Ουσιαστικά, o wrapper παίρνει τις εξόδους του Robosuite, τις φιλτράρει, τις αλλάζει σχήμα και τις σερβίρει στον RL Agent ακριβώς στην μορφή που τις περίμενε.

Βήματα & Χαρακτηριστικά Υλοποίησης:
1) Observation Refactoring: Το Robosuite επιστρέφει ένα dictionary γεμάτο πληροφορίες. Ο Wrapper πετάει ό,τι δεν χρειαζόμαστε, κρατάει μόνο το agentview_image (την εικόνα της κάμερας) και τη μετατρέπει σε pixels.
2) Axis Transposition: Η κάμερα βγάζει την εικόνα σε μορφή $84 \times 84 \times 3$ (Height, Width, Channels). Τα νευρωνικά δίκτυα της PyTorch όμως απαιτούν τη μορφή $3 \times 84 \times 84$ (Channels, Height, Width). Ο Wrapper αλλάζει live τους άξονες της εικόνας σε κάθε frame.
3) Action Space Mapping: Ορίζει στον αλγόριθμο ότι το ρομπότ Panda δέχεται 7 συνεχείς τιμές δράσης, αυστηρά περιορισμένες μεταξύ $-1.0$ και $1.0$, ώστε το RL να μην ζητήσει ποτέ μια "τρελή" ή αδύνατη κίνηση. 
4) Reward & Done Logging - Gym API Compliant: Μετατρέπει το παλιό format επιστροφής του Robosuite στην επίσημη πεντάδα του σύγχρονου Reinforcement Learning (obs, reward, terminated, truncated, info).

Το script θα είναι δομημένο σε 4 βασικά κομμάτια:
1) __init__(self, env): Δέχεται το έτοιμο περιβάλλον του robosuite. Εδώ ορίζουμε το action_space και το observation_space με βάση τα standard του Gym.  
2) obs_space_refactor(self, robosuite_obs): Μια εσωτερική βοηθητική συνάρτηση που παίρνει το dictionary του robosuite, απομονώνει την εικόνα, της κάνει κανονικοποίηση/transpose και την επιστρέφει στην κατάλληλη μορφή.
3) reset(self, seed=None, options=None): Καλεί την env.reset() του robosuite, περνάει το observation από την _process_obs και επιστρέφει το αρχικό state.
4) step(self, action): Δέχεται τη δράση του RL agent, την εκτελεί στο robosuite μέσω της env.step(action), επεξεργάζεται το νέο observation και επιστρέφει την τυποποιημένη πεντάδα.

Επιπλέον, έχει φτιαχτεί ένα script για test του wrapper (test_wrapper.py) το οποίο ελέγχει το Action Space να είναι Box(-1.0, 1.0, (7,), float32), το Observastion Space να δίνει Box(0, 255, (3, 84, 84), uint8) και το Reset Obs Shape να είναι ακριβώς (3,84,84).