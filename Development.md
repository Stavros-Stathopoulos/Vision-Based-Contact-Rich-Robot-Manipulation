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

> **Description:** Το PART 3 εισάγει τον Agent στο πλαίσιο της αυτόνομης Ενισχυτικής Μάθησης (Reinforcement Learning). Εδώ, το ρομπότ σταματά να βασίζεται σε έτοιμα παραδείγματα και μαθαίνει αυτόνομα μέσω δοκιμής και σφάλματος (trial and error) με στόχο τη μεγιστοποίηση των rewards. Λόγω της απουσίας προνομιακών πληροφοριών, η πολιτική εκπαιδεύεται end-to-end να επεξεργάζεται raw pixels και να εξάγει δράσεις ελέγχου.

### Part3_1 : Setup RL Environment & Wrapper
* **Λειτουργία:** Μετατροπή του περιβάλλοντος του robosuite ώστε να είναι πλήρως συμβατό με το standard **Gym/Gymnasium API** που απαιτούν οι σύγχρονες βιβλιοθήκες (Stable-Baselines3). 
* **Χαρακτηριστικά Υλοποίησης:**
  1. **Observation Refactoring:** Απομονώνεται μόνο η εικόνα `agentview_image` και πετιέται η προνομιακή πληροφορία.
  2. **Axis Transposition:** Μετατροπή του frame από μορφή MuJoCo ($84 \times 84 \times 3$) σε μορφή PyTorch ($3 \times 84 \times 84$) live σε κάθε βήμα.
  3. **Action Space Mapping:** Περιορισμός των 7 συνεχών δράσεων αυστηρά στο διάστημα $[-1.0, 1.0]$.
  4. **Gym API Compliance:** Η `step()` και η `reset()` επιστρέφουν πλέον την τυποποιημένη πεντάδα: `(obs, reward, terminated, truncated, info)`.
* **Έλεγχος:** Το script `test_wrapper.py` επιβεβαίωσε την ορθότητα των spaces και των shapes (`uint8` array διάστασης `(3, 84, 84)`).

### Part3_2 : Train Reinforcement Learning Agent

* **Λειτουργία:** Εκπαίδευση αυτόνομου Agent με τον αλγόριθμο **SAC (Soft Actor-Critic)** χρησιμοποιώντας `CnnPolicy` για την end-to-end επεξεργασία των εικόνων και `reward_shaping=True` για την καθοδήγηση του ρομπότ.
* **Μαθηματικές Ενδείξεις Logs (ανά 400 βήματα):**
  * `rollout/ep_rew_mean`: Σταδιακή άνοδος του reward (σωστή συμπεριφορά εξερεύνησης).
  * `train/critic_loss`: Πτώση και σταθεροποίηση του σφάλματος του Critic.
  * `train/ent_coef`: Αυτόματη μείωση της εντροπίας (*entropy tuning*), δείχνοντας ότι ο Agent περνάει σταδιακά από το *exploration* στοχευμένα στο *exploitation*.

#### CRITICAL ISSUE: CPU Memory Leak & Hardware Constraints
* **Το Πρόβλημα:** Λόγω έλλειψης αποκλειστικής κάρτας γραφικών (GPU), η εκπαίδευση εκτελείται αποκλειστικά στην CPU (`Using cpu device`), καθιστώντας τη διαδικασία forward/backward των frames εξαιρετικά αργή (3-4 FPS). Επιπλέον, λόγω low-level memory leak της MuJoCo κατά το reset των εικόνων και του μεγάλου replay buffer, η RAM υπερφορτωνόταν και ο κώδικας κράσαρε με `Could not allocate memory` περίπου στα 3.100 timesteps, εμποδίζοντας την ολοκλήρωση του Block των 5.000 βημάτων.
* **Στρατηγική Επίλυσης (Hard Reset Loop & CPU Optimization):**
  Αντί για απλό reset, ο κώδικας στο `train_rl.py` αναδιαμορφώθηκε ριζικά:
  1. **Iterative Block Training:** Η εκπαίδευση έσπασε σε πολύ μικρά και ελεγχόμενα blocks των **1.000 timesteps** (`STEPS_PER_BLOCK = 1000`) με απενεργοποιημένο το reset του global step (`reset_num_timesteps=False`).
  2. **Hard Environment Purge:** Στο τέλος κάθε block, το περιβάλλον κλείνει οριστικά με `env.close()`, αναγκάζοντας το σύστημα να καταστρέψει τα low-level C++ αντικείμενα και τα off-screen buffers της MuJoCo, μηδενίζοντας τη διαρροή.
  3. **Memory Flush & Reconnect:** Εκτελείται καθολικό `gc.collect()`, απελευθερώνεται η heap μνήμη μέσω `ctypes`, δημιουργείται ένα ολοκαίνουργιο instance περιβάλλοντος και συνδέεται live με τον υπάρχοντα SAC Agent μέσω της `model.set_env(env)`.
  4. **Buffer Reduction:** Το `buffer_size` μειώθηκε στις 2.000 εικόνες και το `batch_size` στο 32 για να προστατευθεί η RAM της CPU και να επιταχυνθούν τα μαθηματικά updates.