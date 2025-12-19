import pandas as pd

def generate_golden_test_data():
    # 0 = Safe, 1 = Unsafe
    data = [
        # --- SECTION 1: HARD CUSS WORDS (The "Bouncer" should catch these) ---
        ["Tu bsdk hai kya?", 1],
        ["Mc bc mat kar yaha par", 1],
        ["He is a chutiya person", 1],
        ["Bhenchod dimag kharab mat kar", 1],
        ["Ye jagah randikhana bana rakha hai", 1],
        ["Madarchod police wale kuch nahi karte", 1], # Cuss word present -> Unsafe regardless of context
        ["Stop acting like a gandu", 1],
        ["Saala harami hai wo shopkeeper", 1],

        # --- SECTION 2: SUBTLE THREATS / HARRASSMENT (The "AI" must catch these) ---
        ["Tu bahar mil mujhe, batata hu", 1],  # No cuss word, but threat
        ["Waha ladkiyon ko chedna common hai", 1],
        ["He was following me for 10 minutes", 1],
        ["Ajeeb tarah se ghoor raha tha wo uncle", 1],
        ["Touching and groping happens in the crowd", 1],
        ["Avoid this place at night, very shady people", 1],
        ["Ladke comment pass karte hai", 1],
        ["I felt very uncomfortable around him", 1],
        ["Rate kya hai tera?", 1], # Contextual harassment
        ["Gandi nazron se dekh raha tha", 1],
        ["He tried to grab my hand", 1],
        ["Not safe for women, trust me", 1],

        # --- SECTION 3: TRICKY SAFE CONTEXT (Ambiguous words used safely) ---
        ["Mera kutta (dog) bohot cute hai", 0], # 'Kutta' usually insult, here animal
        ["Police ne usko mara (hit) kyunki wo chori kar raha tha", 0], # Violence but Justice
        ["We attended a seminar on anti-harassment laws", 0], # 'Harassment' word used factually
        ["Harassing women is a crime", 0], # Statement of fact
        ["Don't be afraid, the guard is here", 0], # 'Afraid' used positively
        ["Murder mystery movies are my favorite", 0], # 'Murder' in entertainment context
        ["Kill dil is a bollywood movie", 0], # 'Kill' in title
        ["She is fighting for her rights", 0], # 'Fighting' used metaphorically
        ["Report any abuse to the helpline", 0], 

        # --- SECTION 4: STANDARD SAFE REVIEWS (English & Hinglish) ---
        ["This place is amazing for families", 0],
        ["Bhai yaha ka khana mast hai", 0],
        ["Lighting achi hai, raat ko safe feel hota hai", 0],
        ["Security guards are always present", 0],
        ["Best place to hang out with friends", 0],
        ["Metro station is walking distance, very convenient", 0],
        ["Crowd is decent, no chapris", 0],
        ["Police patrolling hoti rehti hai", 0],
        ["Very clean and well maintained park", 0],
        ["Uncle dukan wale bohot helpful hai", 0],
        ["Loved the vibe of this place", 0],
        
        # --- SECTION 5: ENGLISH TOXICITY ---
        ["You are an idiot", 1],
        ["Shut up you moron", 1],
        ["I will slap you", 1],
        ["Creepy guys everywhere", 1],
        ["This place is hell", 1],
        
        # --- SECTION 6: MORE HINGLISH VARIETY ---
        ["Ek dum bakwas jagah hai", 1], # Negative review (could be 1 or 0 depending on app logic, usually Unsafe if hostile)
        ["Maza aa gaya yaha aake", 0],
        ["Dar lagta hai yaha", 1],
        ["Koi help nahi karta yaha", 1],
        ["Sab log friendly hai", 0]
    ]

    df = pd.DataFrame(data, columns=['text', 'actual_label'])
    df.to_csv("test_dataset.csv", index=False)
    print("✅ Generated 50 Test Rows in 'test_dataset.csv'")

if __name__ == "__main__":
    generate_golden_test_data()