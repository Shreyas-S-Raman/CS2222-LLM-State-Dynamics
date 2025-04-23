import random 


def gen_prompts(entity_count):
    all_names = ["Dean", "Sarah", "Jack", "Kate", "Ryan", "Bob", "Jill", "Mary", "Peter", "Luke", "John",
             "Sean", "Elle", "Neil", "Tom", "Jerry", "Joan", "Blake", "Seth", "Beth", "Kelly", "Ben",
             "Jean"]
    all_fruits = ["apple", "pear", "peach", "grape", "mango", "banana", "kiwi", "orange", "lemon",
              "melon", "cherry", "plum", "fig", "apricot", "durian", "honeydew", "cantaloupe", "papaya",
              "persimmon", "lemon", "lime", "lychee", "guava", "coconut"]
    data = [] # full data

    for ec in range(entity_count):
        # get shuffled aval names
        aval_names = all_names[:ec]
        aval_fruit = all_fruits[:ec]
        random.shuffle(aval_names)
        random.shuffle(aval_fruit)

        base_prompt = []
        base_prompt.append(f"{', '.join(aval_names)} walk into a fruit store. There are only {ec} fruits {', '.join(aval_fruit)}. Each person gets a different fruit.")
        
        # start from 1 clue, keep cumulating
        for clue_num in range(1, ec):
            name_to_fruit = {}
            
            choice = random.choice([0, 1])
            n1 = random.choice(range(len(aval_names)))
            name = aval_names[n1]
            fruit = aval_fruit[n1]

            aval_names.pop(n1)
            aval_fruit.pop(n1)

            if choice:
                clue = f"{name} gets the {fruit}."
            else:
                n2 = random.choice(range(len(aval_names)))
                clue = f"{aval_names[n2]} gives {name} the {fruit}."
            
            base_prompt.append(clue)

            # now output just this clue_num prompt
            full_prompt = base_prompt[:]
            n3 = random.choice(range(len(aval_names)))
            full_prompt.append(f"{aval_names[n3]} can have the")
            full_prompt = " ".join(full_prompt)
            answer = ",".join(aval_fruit)

            data.append({
                "entities" : ec,
                "clues" : clue_num,
                "prompt" : full_prompt,
                "answer" : answer,
            })
    return data


data = gen_prompts(22)
print(data)
            





