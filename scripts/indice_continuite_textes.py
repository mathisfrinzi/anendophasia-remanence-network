# Indice de continuite sur 5 textes de 100 tokens. A lancer depuis la racine : python scripts/indice_continuite_textes.py
import sys, torch
sys.path.insert(0, "src")
from tokenizers import Tokenizer
import s2_models as M
tok = Tokenizer.from_file("tokenizer/bpe_4096.json")
S = [
 "Once upon a time, there was a little girl named Lily. She loved to play in the garden with her cat. One sunny day, she saw a big red ball under the tree. She picked it up and threw it to her cat. The cat jumped and caught the ball in the air. Lily laughed and clapped her hands. Then her mom called her for lunch. They ate soup and bread together. After lunch, Lily and her cat took a nap in the warm sun. It was a very happy day for them both.",
 "Tom was a boy who liked to build things. He had a box full of blocks in his room. Every morning, he made a tall tower and a small house. One day, his little sister came in and wanted to help. Tom said no at first, but then he smiled. They built a big castle together. It had a door, a window, and a flag on top. When they were done, they showed it to their dad. Dad said it was the best castle he had ever seen. Tom and his sister were proud.",
 "The sun went down and the sky turned pink and orange. A small bird sat on a branch and looked at the town below. It was tired after a long day of flying. The bird wanted to find a warm place to sleep. It saw a little house with a open window and flew inside. A kind old woman gave it some seeds and water. The bird sang a soft song to say thank you. The woman put a blanket near the window. That night, the bird slept well and dreamed of the blue sky.",
 "Mia and Sam went to the park with their grandma. They saw a pond with ducks swimming in it. Mia wanted to feed the ducks, so grandma gave her some bread. Sam ran to the swings and began to swing high. A big dog came running and barked at the ducks. The ducks quacked and swam away fast. Grandma took the dog's leash and said sorry to the kids. The dog licked Mia's hand and wagged its tail. They all laughed and sat down to eat a snack under a big tree.",
 "Ben found a shiny key in the sand at the beach. He wondered what it could open, so he showed it to his friend Anna. They looked all over the beach for a box or a door. Near the rocks, they found a small old chest half hidden in the sand. Ben put the key in the lock and turned it. The chest opened with a click. Inside there were colorful shells and a note. The note said to share the shells with everyone. Ben and Anna gave a shell to each child on the beach."]
ids = [tok.encode(s).ids[:100] for s in S]
print([len(i) for i in ids])
x = torch.tensor(ids)
for name in ("V_lam0", "V_lam1_ext"):
    ck = torch.load("results/s2_%s_model.pt" % name, map_location="cpu", weights_only=False)
    m = M.build(ck["spec"], ck["V"], ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        _, fr = m(x, return_frames=True)
    g = fr.reshape(*fr.shape[:2], 2, 3, 8, 8)
    f, s = (M.TextImageAE.continuity_index(g[:, :, i]).item() for i in (0, 1))
    print(name, "indice rapide %.2f  lent %.2f  moyenne %.2f" % (f, s, (f + s) / 2))
