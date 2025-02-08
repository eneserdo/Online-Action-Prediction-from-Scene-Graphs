import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

COAX_OBJECTS = ["screwdriver", "hose", "valve", "valve_terminal", "box_w_screws", 
           "box_w_membrane", "soldering_station", "soldering_iron", "soldering_tin", 
           "soldering_board", "capacitor", "robot", "human", "RightHand", "eef_robot", 
           "robot_base"]

COAX_OBJECTS_EXPANDED = ["screwdriver", "hose", "valve", "valve terminal", "box with screws", 
                 "box with membrane", "soldering station", "soldering iron", "soldering tin", 
                 "soldering board", "capacitor", "robot", "human", "RightHand", "eef robot", 
                 "robot base"]

COAX_ACTIONS = ['approach', 'grab_screwdriver', 'plug_screwdriver', 'grab_valve', 
                'screw_screwdriver', 'release_valve', 'grab_soldering_iron', 'plug_soldering_iron', 
                'retreat', 'join_screwdriver', 'grab_valve_terminal', 'plug_valve_terminal', 
                'place_screwdriver', 'grab_box_w_screws', 'place_box_w_screws', 'grab_hose', 
                'wait_for_robot', 'plug_hose', 'grab_box_w_membrane', 'grab_soldering_station', 
                'solder_hose', 'release_soldering_station', 'release_box_w_membrane']

COAX_ACTIONS_EXPANDED = ['approach', 'grab screwdriver', 'plug screwdriver', 'grab valve', 
           'screw screwdriver', 'release valve', 'grab soldering iron', 'plug soldering iron', 
           'retreat', 'join screwdriver', 'grab valve terminal', 'plug valve terminal', 
           'place screwdriver', 'grab box with screws', 'place box with screws', 'grab hose', 
           'wait for robot', 'plug hose', 'grab box with membrane', 'grab soldering station', 
           'solder hose', 'release soldering station', 'release box with membrane']


BIMACS_OBJECTS = ['bowl', 'knife', 'screwdriver', 'cuttingboard', 'whisk', 'hammer', 'bottle', 'cup',
           'banana', 'cereals', 'sponge', 'woodenwedge', 'saw', 'harddrive', 'LeftHand',
           'RightHand']


BIMACS_ACTIONS = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut', 'hammer', 'saw',
           'stir', 'screw', 'drink', 'wipe']


RELATIONS = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
             'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
             'getting close', 'moving apart', 'stable']



def main():
    n_components = 16

    # map all names to their expanded names
    item_mapping = {}

    item_mapping.update({name: name for name in BIMACS_OBJECTS})
    item_mapping.update({name: expanded for name, expanded in zip(COAX_OBJECTS, COAX_OBJECTS_EXPANDED)})
    item_mapping.update({name: name for name in RELATIONS})


    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    ref_embeddings = model.encode(list(item_mapping.values()))


    print(f"PCA {n_components=}")
    pca = PCA(n_components=n_components)
    pca.fit(ref_embeddings)

    ref_embeddings_pca = pca.transform(ref_embeddings)

    # save as named numpy array, with keys as the item names
    np.savez("embeddings.npz", **{item: ref_embeddings_pca[i] for i, item in enumerate(item_mapping.keys())})

    recovered_embeddings = pca.inverse_transform(ref_embeddings_pca)
    error = np.mean((ref_embeddings - recovered_embeddings)**2)
    exp_var = pca.explained_variance_.sum()

    # Compute pairwise cosine similarity
    original_similarity = cosine_similarity(ref_embeddings)
    reduced_similarity = cosine_similarity(ref_embeddings_pca)

    # Compute correlation between original and reduced similarities
    correlation = np.corrcoef(original_similarity.flatten(), reduced_similarity.flatten())[0, 1]

    print(f"Error: {error:.5f}")
    print(f"Explained variance: {exp_var:.5f}")
    print(f"Similarity preservation correlation: {correlation:.5f}")


if __name__ == "__main__":
    main()
