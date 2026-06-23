# Project Overview & Latest Implementation Changes

---

## PART 1: Foundations & Architecture

> **Description:** Το PART 1 θέτει τις προγραμματιστικές και μαθηματικές βάσεις του project. Εστιάζει στην αρχικοποίηση του προβλήματος ως Markov Decision Process (MDP) μέσα από το robosuite, στον ορισμό της διεπαφής (Interface) της πολιτικής ελέγχου, και στην κατασκευή ενός CNN Vision Encoder (Nature CNN) ικανού να μετατρέπει raw pixels εικόνας σε low-dimensional embeddings, χωρίς τη χρήση privileged πληροφορίας.

### Part1_1 : test_robosuite.py
* **Λειτουργία:** Το script αυτό αρχικοποιεί ακριβώς το *Markov Decision Process (MDP)*, εκτελεί τη συνάρτηση `env.step(action)` και επιστρέφει το επόμενο observation και reward.
* **Θεωρητική Σύνδεση:** Εφαρμόζει στην πράξη τον θεμελιώδη ορισμό της Ενισχυτικής Μάθησης και τον κύκλο αλληλεπίδρασης: *Agent-Environment* ($State \ s_t, Action \ a_t, Reward \ r_{t+1}$).

### Part1_2 : base_controller.py & test_interface.py
* **Λειτουργία:** Η Πολιτική (*Policy - $\pi$*) ορίζεται ως μια συνάρτηση που παίρνει μια κατάσταση και επιστρέφει μια κίνηση: $a = \pi(s)$ ή $a = \pi(\cdot|s)$.
* **Θεωρητική Σύνδεση:** Το Interface που υλοποιήθηκε με τη μέθοδο `act(obs)` είναι η άμεση μαθηματική και προγραμματιστική αποτύπωση αυτής της πολιτικής $\pi(s)$, έτοιμη να υποδεχθεί Baseline, Behavior Cloning, ή RL προσεγγίσεις.

### Part1_3 : CNN_Vision_Encoder.py
* **Λειτουργία:** Όταν ο χώρος καταστάσεων είναι πολύ μεγάλος ή συνεχής (όπως μια εικόνα pixel $84 \times 84 \times 3$), είναι αδύνατο να χρησιμοποιηθούν look-up tables. Χρειαζόμαστε έναν προσεγγιστή συναρτήσεων (*Function Approximator*).
* **Χαρακτηριστικά Αρχιτεκτονικής:** * Επιλέχθηκε η αρχιτεκτονική του *Nature CNN* με κατάλληλη διόρθωση των φίλτρων (μικρότερα strides) για τη διατήρηση της χωρικής ανάλυσης (*spatial resolution*).
  * **Διατήρηση Χώρου:** Το τελικό feature map πριν το `Flatten` είναι $17 \times 17$, δίνοντας στο δίκτυο $18.496$ τοπικά χαρακτηριστικά. Αυτό επιτρέπει στη `Linear` layer να συσχετίσει τις άκρες της τσιμπίδας με το παξιμάδι.
  * Εξάγει γεωμετρικά χαρακτηριστικά χωρίς privileged πληροφορία, μετατρέποντας τα pixels σε ένα χρήσιμο low-dimensional embedding $256$ διαστάσεων, έτοιμο για gradient updates (*backpropagation*).

---

## PART 2: Imitation Learning (Behavior Cloning)

> **Description:** Το PART 2 υλοποιεί τη μέθοδο του Imitation Learning (Μίμηση Συμπεριφοράς). Αρχικά αναπτύσσεται ένας προγραμματισμένος Heuristic Expert (Finite State Machine) που κινείται στον χώρο για τη δημιουργία ενός minimum performance benchmark. Στη συνέχεια, συλλέγεται ένα dataset επιτυχημένων τροχιών (raw εικόνες και continuous actions) το οποίο χρησιμοποιείται για την εκπαίδευση (Supervised Learning / MSE Loss) και την live αξιολόγηση μιας Behavior Cloning πολιτικής.

### Part2_1 : Develop Simple Baseline Controller (Scripted Heuristic Policy)
* **Λειτουργία:** Σχεδιάστηκε ένας *Finite State Machine (FSM)* που εκτελεί μια ακολουθία κινήσεων στον συνεχή χώρο δράσης (*Continuous Action Space*) του Panda, ο οποίος ελέγχεται ως εξής:
  1. `action[0:3]`: Γραμμική ταχύτητα/μετατόπιση στον άξονα (X, Y, Z).
  2. `action[3:6]`: Περιστροφή (Roll, Pitch, Yaw).
  3. `action[6]`: Έλεγχος Gripper (Τσιμπίδα): $-1.0$ για άνοιγμα, $+1.0$ για κλείσιμο.
* **Στόχος Baseline:** Κληρονομεί από τον `BaseController` και εκτελεί την τυφλή ακολουθία: *Προσέγγιση $\rightarrow$ Χαμήλωμα $\rightarrow$ Κλείσιμο Gripper $\rightarrow$ Ανύψωση*. Λειτουργεί ως το τέλειο κατώτατο όριο σύγκρισης (*minimum benchmark*) για να εκτιμηθεί η αξία των RL αλγορίθμων που βλέπουν εικόνα ($use\_object\_obs=False$).

### Part2_2 : Collect Demonstration Dataset
* **Λειτουργία:** Στόχος είναι η συλλογή ενός συνόλου δεδομένων από έμπειρες τροχιές $\mathcal{D} = \{(s_t, a_t)\}_{t=1}^N$ για Supervised Learning, αποθηκεύοντας τις raw εικόνες (`agentview_image`) ως καταστάσεις $s_t$ και τα continuous vectors ως δράσεις $a_t$ σε συμπιεσμένο αρχείο `.npz`.
* **CHANGES:** Εμφανιζόταν `ValueError: Error: engine error: Could not allocate memory` επειδή η `env.reset()` αναδομούσε ολόκληρο το simulation από το XML string, προκαλώντας *memory leak*. Εφαρμόστηκε εξαναγκασμένο **Garbage Collection (`gc.collect()`)** για καθαρισμό της μνήμης ανά επεισόδιο και προστέθηκε όριο `MAX_TOTAL_EPISODES = 100`.

### Part2_3 : Train Behavior Cloning Policy
* **Λειτουργία:** Εκπαίδευση πολιτικής $\pi_{\theta}(s)$ με Supervised Learning. Ο `CNNEncoder` επεξεργάζεται τις εικόνες και ένα Fully Connected layer εξάγει τις 7 continuous actions, ελαχιστοποιώντας το Mean Squared Error (MSE) Loss:
$$\min_{\theta} \sum_i \|\pi_{\theta}(s_i) - a_i\|^2$$
* **Αποτέλεσμα:** Το script `train_bc.py` φορτώνει το dataset, εκτελεί το optimization και αποθηκεύει τις έτοιμες γνώσεις (βάρη) στο αρχείο `bc_model.pth`.

### Part2_4 : Evaluate Imitation Learning Policy
* **Λειτουργία:** Live αξιολόγηση του AI μοντέλου στο περιβάλλον `NutAssembly`. Φορτώνονται τα βάρη από το `bc_model.pth` σε `eval()` mode και ελέγχεται αν το ρομπότ γενικεύει σωστά πάνω στα pixels.
* **Διαδικασία:** Τρέχει συγκεκριμένα δοκιμαστικά επεισόδια (Evaluation Loop), εκτελεί live inference ($a = \pi_{\theta}(s)$) και υπολογίζει το τελικό **Success Rate %** (ποσοστό επεισοδίων με $Reward > 0$). Ενσωματώθηκε Garbage Collection ανά επεισόδιο για την αποφυγή κρασαρίσματος της MuJoCo.

---

## PART 3: Reinforcement Learning (Autonomous Control)

> **Description:** Το PART 3 υλοποιεί τη μετάβαση από την επιβλεπόμενη μίμηση στην αυτόνομη εξερεύνηση, τη διαχείριση πόρων και τη βελτιστοποίηση της πολιτικής ελέγχου σε συνεχή χώρο δράσεων (Continuous Action Space) για το task του Nut Assembly.

### Task 3.1: Setup RL Environment & Wrapper
* **Λειτουργία:** Μετατροπή του περιβάλλοντος του robosuite ώστε να είναι πλήρως συμβατό με το standard **Gym/Gymnasium API** που απαιτούν οι σύγχρονες βιβλιοθήκες (Stable-Baselines3). 
* **Χαρακτηριστικά Υλοποίησης:**
  1. **Observation Refactoring:** Απομονώνεται μόνο η εικόνα `agentview_image` (διάστασης $84 \times 84 \times 3$) από την εικονική off-screen κάμερα και πετιέται η προνομιακή πληροφορία.
  2. **Axis Transposition:** Μετατροπή του frame από μορφή MuJoCo (`H, W, C`) σε μορφή PyTorch (`C, H, W`) live σε κάθε βήμα.
  3. **Action Space Mapping:** Περιορισμός και κανονικοποίηση των 7 συνεχών δράσεων αυστηρά στο διάστημα $[-1.0, 1.0]$.
* **Έλεγχος:** Το script `test_wrapper.py` επιβεβαίωσε την ορθότητα των spaces και των shapes (`uint8` array διάστασης `(3, 84, 84)`).

### Task 3.2: Train RL-Agent (Recovery Logic for Contact-Rich Failures)
* **Λειτουργία:** Εκπαίδευση αυτόνομου Agent με τον αλγόριθμο **SAC (Soft Actor-Critic)** χρησιμοποιώντας `CnnPolicy` για την end-to-end επεξεργασία των εικόνων και `reward_shaping=True` για την καθοδήγηση του ρομπότ.
* **CRITICAL HARDWARE ISSUE & RECOVERY LOGIC:**
  Λόγω έλλειψης αποκλειστικής κάρτας γραφικών (GPU) και low-level memory leak της MuJoCo κατά το off-screen rendering των καμερών, η RAM υπερφορτωνόταν και ο κώδικας κράσαρε με `Could not allocate memory` περίπου στα 3.100 timesteps. Σχεδιάστηκε ένας εξελιγμένος μηχανισμός ανάκτησης πόρων (**Iterative Block Training - IBT**):
  1. **Block Allocation:** Η εκπαίδευση σπάει σε ελεγχόμενα Blocks των **2.000 timesteps** (`STEPS_PER_BLOCK = 2000`) με `reset_num_timesteps=False` για τη διατήρηση των global steps και των Adam optimizers.
  2. **Hard Environment Purge:** Στο τέλος κάθε Block, εκτελείται `env.close()` για την καταστροφή των low-level C++ rendering buffers, γίνεται χειροκίνητο `gc.collect()`, και εξαναγκάζεται εκκαθάριση της heap μνήμης μέσω C-bindings: `ctypes.CDLL(None).malloc_trim(0)`.
  3. **Memory Optimization:** Ενεργοποιήθηκε η παράμετρος `optimize_memory_usage=True` στον SAC, η οποία δημιουργεί δυναμικά τις επόμενες καταστάσεις, μειώνοντας την κατανάλωση RAM του Replay Buffer κατά 50%.
  4. **Parallelization:** Ορίστηκε `torch.set_num_threads(os.cpu_count())` για πλήρη παράλληλη επεξεργασία πινάκων στην CPU.

### Task 3.3: Apply Domain Randomization for Robustness
* **Υλοποίηση & Proof of Concept (PoC):** Για την επαλήθευση της σταθερότητας του pipeline, εκτελέστηκε ένα Nominal Run **10.000 timesteps** (5 Blocks των 2.000 βημάτων). Τα αποτελέσματα έδειξαν υποδειγματική μαθηματική σύγκλιση:
  * **Entropy Tuning (`ent_coef`):** Μειώθηκε ομαλά από το `0.914` στο **`0.0526`**, επιβεβαιώνοντας ότι ο Agent απέκτησε σιγουριά, περνώντας από το exploration στο exploitation.
  * **Actor Loss:** Μειώθηκε σταθερά από το `-10.8` στο **`-29.4`**, δείχνοντας ότι ο Actor έμαθε να παράγει τροχιές με υψηλότερα Q-values.
  * **Reward (`ep_rew_mean`):** Σταθεροποιήθηκε στο **`0.24 - 0.28`**, αποδεικνύοντας ότι ο Agent έμαθε να οδηγεί σταθερά τον Panda gripper πολύ κοντά στο παξιμάδι.
* **Στρατηγική Robustness (Full Production Run):** Επειδή τα contact-rich tasks απαιτούν εκτενή έκθεση σε διαφορετικές αρχικές θέσεις του παξιμαδιού (Domain Randomization), το pipeline είναι 100% έτοιμο να εκτελέσει το **Full Run των 20.000 - 50.000+ timesteps** συνεχίζοντας από το παραχθέν checkpoint, ώστε ο Agent να αναπτύξει τη στιβαρότητα που απαιτείται για το τελικό insertion.

* **Artifact Serialization:** Το μοντέλο αποθηκεύτηκε επιτυχώς ως `sac_nut_assembly.zip` περιλαμβάνοντας τα βάρη των δικτύων (`policy.pth`) και τις πλήρεις καταστάσεις των optimizers για απρόσκοπτη μελλοντική συνέχιση.

### Task 3.4: Make Diagramms & Simulation to Test the RL-Agent
* **Λειτουργία:** Υλοποίηση του script `src/models/test_rl.py`. Φορτώνει το εκπαιδευμένο μοντέλο, παρακάμπτει το interactive backend του matplotlib χρησιμοποιώντας `matplotlib.use('Agg')` για την αποφυγή σφαλμάτων του `tkinter` στα Windows, εκτελεί deterministic επεισόδια και παράγει αυτόματα διαγράμματα.
* **Αξιοποίηση στο Project:** Παράγει το διάγραμμα `rl_logs/evaluation_live_plot.png` αποτυπώνοντας τα rewards ανά επεισόδιο, επιβεβαιώνοντας ποσοτικά και ποιοτικά τη λειτουργικότητα του pipeline.

---

## PART 4: Evaluation & Deployment

> **Description:** Το PART 4 αφορά την ποιοτική και ποσοτική αξιολόγηση της εκπαιδευμένης πολιτικής. Μέσα από αυτοματοποιημένα scripts, μετατρέπουμε τα binary αρχεία των βαρών σε διαγράμματα σύγκλισης και βίντεο προσομοίωσης, εξασφαλίζοντας τον πλήρη έλεγχο της συμπεριφοράς του ρομπότ.

### Part4_1 : Create Evaluation and Logging Scripts
* **Λειτουργία:** Μέσω του `test_rl.py` (με ενεργοποιημένο το `has_renderer=True`), ανοίγει live το 3D graphical παράθυρο της MuJoCo στα Windows. Επιτρέπει την οπτική επιβεβαίωση της κίνησης του Panda, αποδεικνύοντας ότι ο βραχίονας κατευθύνεται με ακρίβεια προς το αντικείμενο βασιζόμενος μόνο στην όραση.

### Part4_2 : Record and Edit Demo Video
* **Λειτουργία:** Υλοποίηση του script `src/models/evaluate_and_record.py`. Χρησιμοποιεί τον off-screen renderer της MuJoCo για να καταγράψει off-screen τα frames υψηλής ανάλυσης της κάμερας `agentview` και να συνθέσει αυτόματα ένα αρχείο βίντεο `evaluation_simulation.mp4`.
* **Αξιοποίηση στο Project:** Αποτελεί το τελικό παραδοτέο βίντεο της εργασίας, αναδεικνύοντας τις nominal επιτυχίες προσέγγισης, καθώς και τη συμπεριφορά του Agent κατά τις force interactions.
