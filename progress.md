# Repository Issues & Tasks Mapping

Το αρχείο αυτό αποτυπώνει πλήρως τη χαρτογράφηση όλων των επίσημων Tasks (Issues) του repository για το project **Vision-Based Contact-Rich Robot Manipulation**. Για κάθε Task περιγράφεται η συγκεκριμένη υλοποίηση στον κώδικα, ο σκοπός του και ο τρόπος αξιοποίησής του στο τελικό σύστημα.

---

## 🟢 MILESTONE 1: Foundations & Architecture

### Task 1.1: Environment Setup and robosuite Testing
* **Αντιστοίχιση στον Κώδικα:** `main.py`, `src/__init__.py`
* **Σκοπός & Χρησιμότητα:** Αρχικοποιεί το περιβάλλον `NutAssembly` του `robosuite` με τον Operational Space Controller (`BASIC`). Απενεργοποιεί πλήρως τις έτοιμες low-level συντεταγμένες (`use_object_obs=False`) και ενεργοποιεί την εικονική κάμερα `agentview` με ανάλυση $84 \times 84$.
* **Αξιοποίηση στο Project:** Αποτελεί την "αρένα" προσομοίωσης και ορίζει τους νόμους της φυσικής (MuJoCo). Εφαρμόζει στην πράξη το Markov Decision Process (MDP) και τον κύκλο αλληλεπίδρασης Agent-Environment.

### Task 1.2: Implement Common Controller Interface
* **Αντιστοίχιση στον Κώδικα:** `src/controllers/base_controller.py`
* **Σκοπός & Χρησιμότητα:** Ορίζει την abstract κλάση `BaseController` και την υποχρεωτική μέθοδο `act(obs)`.
* **Αξιοποίηση στο Project:** Εξασφαλίζει την προγραμματιστική συμβατότητα και ομοιομορφία. Οποιοσδήποτε controller (Heuristic, Behavior Cloning, ή RL) αναπτυχθεί, δέχεται το ίδιο observation και επιστρέφει το ίδιο action vector (7 συνεχείς τιμές), επιτρέποντας την άμεση εναλλαγή τους.

### Task 1.3: Build CNN Encoder for Vision Pipeline
* **Αντιστοίχιση στον Κώδικα:** `src/encoders/cnn_encoder.py`, `src/encoders/preprocessing.py`
* **Σκοπός & Χρησιμότητα:** Υλοποιεί ένα custom *NatureCNN Feature Extractor* (3 Convolutional Layers με ReLU και ένα Linear Head).
* **Αξιοποίηση στο Project:** Αποτελεί τα **"μάτια"** του ρομπότ. Παίρνει τα raw pixels ($84 \times 84 \times 3$) από την κάμερα `agentview`, εξάγει spatial representations (χωρικές αναπαραστάσεις) και τα συμπυκνώνει σε ένα low-dimensional embedding 256 διαστάσεων. Επιτρέπει στον Agent να καταλαβαίνει πού βρίσκεται το παξιμάδι χωρίς privileged πληροφορία.

---

## 🔵 MILESTONE 2: Imitation Learning / Behavior Cloning

### Task 2.1: Develop Simple Baseline Controller
* **Αντιστοίχιση στον Κώδικα:** `src/controllers/baseline_controller/controller.py`
* **Σκοπός & Χρησιμότητα:** Κατασκευάζει έναν heuristic controller βασισμένο σε Finite State Machine (FSM). Εκτελεί μια τυφλή, χρονικά προγραμματισμένη ακολουθία κινήσεων (Προσέγγιση $\rightarrow$ Χαμήλωμα $\rightarrow$ Κλείσιμο Gripper $\rightarrow$ Ανύψωση).
* **Αξιοποίηση στο Project:** Λειτουργεί ως το **minimum performance benchmark** για τη σύγκριση των AI αλγορίθμων, καθώς και ως ο "Expert" (έμπειρος οδηγός) για τη συλλογή των δεδομένων εκπαίδευσης.

### Task 2.2: Collect Demonstration Dataset
* **Αντιστοίχιση στον Κώδικα:** `src/demos/collect_demos.py`
* **Σκοπός & Χρησιμότητα:** Τρέχει τον Heuristic Expert και αποθηκεύει τις επιτυχείς τροχιές (ζεύγη εικόνων και actions) στο συμπιεσμένο αρχείο `expert_demos.npz`. Εδώ επιλύθηκε το πρώτο low-level memory leak με εξαναγκασμένο `gc.collect()` ανά επεισόδιο.
* **Αξιοποίηση στο Project:** Δημιουργεί το dataset εκπαίδευσης $\mathcal{D} = \{(s_t, a_t)\}_{t=1}^N$ που είναι απαραίτητο για τη Supervised Μάθηση.

### Task 2.3: Implement Behavior Cloning (BC)
* **Αντιστοίχιση στον Κώδικα:** `src/models/bc_model.py`, `src/models/train_bc.py`
* **Σκοπός & Χρησιμότητα:** Ορίζει το δίκτυο `BehaviorCloningPolicy` (CNN Encoder + MLP Head με έξοδο `Tanh` για περιορισμό στο $[-1, 1]$) και εκτελεί το optimization χρησιμοποιώντας Mean Squared Error (MSE) Loss και τον Adam Optimizer.
* **Αξιοποίηση στο Project:** Εκπαιδεύει τον Agent να μιμείται τον Expert. Μαθαίνει να χαρτογραφεί απευθείας τα raw pixels της εικόνας στις 7 continuous δράσεις του Panda βραχίονα.

### Task 2.4: Evaluate Imitation Learning Policy
* **Αντιστοίχιση στον Κώδικα:** `src/models/evaluate_bc.py`
* **Σκοπός & Χρησιμότητα:** Φορτώνει τα εκπαιδευμένα βάρη του `bc_model.pth` σε κατάσταση αξιολόγησης (`eval()`) και τρέχει αυτόνομα επεισόδια στον εξομοιωτή.
* **Αξιοποίηση στο Project:** Υπολογίζει το επίσημο **Success Rate %** της μίμησης, ελέγχοντας αν ο Agent μπορεί να γενικεύσει σε νέες τυχαίες θέσεις ή αν αποτυγχάνει λόγω συσσώρευσης σφαλμάτων (compounding errors).

---

## 🟡 MILESTONE 3: Reinforcement Learning (Autonomous Control)

### Task 3.1: Setup RL Environment & Wrapper
* **Αντιστοίχιση στον Κώδικα:** `src/environments/gym_wrapper.py`, `src/environments/test_wrapper.py`
* **Σκοπός & Χρησιμότητα:** Μετατρέπει το custom API του robosuite σε standard Gymnasium API, κάνοντας transpose τις εικόνες σε PyTorch format `(3, 84, 84)` και περιορίζοντας το Action Space αυστηρά στο $[-1.0, 1.0]$.
* **Αξιοποίηση στο Project:** Γεφυρώνει τον εξομοιωτή με τη βιβλιοθήκη `Stable-Baselines3`, καθιστώντας δυνατή την εκπαίδευση με τον state-of-the-art αλγόριθμο **Soft Actor-Critic (SAC)**.

### Task 3.2: Train RL-Agent *(Recovery Logic for Contact-Rich Failures)*
* **Αντιστοίχιση στον Κώδικα:** `src/models/train_rl.py`
* **Σκοπός & Χρησιμότητα:** Υλοποιεί το pipeline αυτόνομης εκπαίδευσης του SAC με `CnnPolicy` και `reward_shaping=True`. Αντιμετωπίζει το κρίσιμο memory leak της MuJoCo στην CPU μέσω της στρατηγικής **Iterative Block Training (IBT)** (εκπαίδευση σε Blocks των 2.000 steps, hard `env.close()`, καθαρισμός RAM με `malloc_trim` και επανασύνδεση με `reset_num_timesteps=False`). Ενεργοποιεί επίσης το `optimize_memory_usage=True` για 50% οικονομία μνήμης.
* **Αξιοποίηση στο Project:** Είναι η καρδιά της αυτόνομης μάθησης μέσω trial-and-error. Ο Agent μαθαίνει να ανακάμπτει από αστοχίες ευθυγράμμισης (contact-rich failures) και να διορθώνει την τροχιά του για να προσεγγίσει το αντικείμενο.

### Task 3.3: Apply Domain Randomization for Robustness
* **Αντιστοίχιση στον Κώδικα:** `src/models/train_rl.py` (Μεταβλητή `TOTAL_TIMESTEPS`)
* **Σκοπός & Χρησιμότητα:** Εξθέτει τον Agent σε εκτεταμένο αριθμό βημάτων (Production Run 20k ή 50k steps) όπου οι αρχικές θέσεις του παξιμαδιού (nut) και του πείρου (peg) αλλάζουν συνεχώς και τυχαία.
* **Αξιοποίηση στο Project:** Αναγκάζει το NatureCNN να αναπτύξει στιβαρότητα (robustness) και να αγνοεί οπτικούς ή γεωμετρικούς θορύβους, εξασφαλίζοντας ότι το ρομπότ θα πετύχει το insertion κάτω από οποιαδήποτε τυχαία αρχική συνθήκη.

### Task 3.4: Make Diagramms & Simulation to Test the RL-Agent
* **Αντιστοίχιση στον Κώδικα:** `src/models/test_rl.py`
* **Σκοπός & Χρησιμότητα:** Φορτώνει το τελικό παραχθέν αρχείο `sac_nut_assembly.zip`, ενεργοποιεί το live graphical παράθυρο των Windows (`has_renderer=True`) και εκτελεί deterministic επεισόδια, αποθηκεύοντας παράλληλα τα rewards στο `rl_logs/evaluation_live_plot.png`.
* **Αξιοποίηση στο Project:** Αποτελεί την **τελική οπτική και ποσοτική επιβεβαίωση** του Milestone 3. Επιτρέπει στην ομάδα να δει live τον βραχίονα Panda να καθοδηγείται από την κάμερα προς το παξιμάδι και παράγει τα διαγράμματα για την τεχνική αναφορά.

---

## 🔴 MILESTONE 4: Reporting & Deliverables

### Task 4.1: Create Evaluation and Logging Scripts
* **Αντιστοίχιση στον Κώδικα:** `src/models/test_rl.py` (Ενότητα Plotting / Evaluation Loop)
* **Σκοπός & Χρησιμότητα:** Σταθεροποιεί την καταγραφή των μετρικών απόδοσης (Total Reward, Steps ανά επεισόδιο) χωρίς interactive κρασαρίσματα backend.

### Task 4.2: Record and Edit Demo Video
* **Αντιστοίχιση στον Κώδικα:** `src/models/evaluate_and_record.py` (Off-screen render loop)
* **Σκοπός & Χρησιμότητα:** Χρησιμοποιεί τον off-screen renderer της MuJoCo για να αποθηκεύσει τις κινήσεις του ρομπότ σε αρχείο βίντεο `.mp4`. Θα αποτυπώσει τις nominal επιτυχίες και τις στρατηγικές ανάκτησης από failures.

### Task 4.3: Write the Final Technical Report
* **Αντιστοίχιση στον Κώδικα:** `DEVELOPMENT.md`, Τεχνική Αναφορά (Word/PDF)
* **Σκοπός & Χρησιμότητα:** Η τελική επιστημονική τεκμηρίωση της εργασίας. Αναλύει τη μαθηματική σύγκλιση (μείωση του `ent_coef`, συμπεριφορά του `actor_loss`) και επεξηγεί τις μηχανικές λύσεις που δόθηκαν στα hardware constraints της CPU.