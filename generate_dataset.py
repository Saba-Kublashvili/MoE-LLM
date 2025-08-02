import json
import random
from faker import Faker

fake = Faker()

def create_multistep_problem():
    """Generates a 2-step or 3-step arithmetic word problem."""
    
    # Define problem templates
    templates = [
        {
            "type": "2-step",
            "prompt": "{name1} starts with {n1} {item}. They buy {n2} more, then give away {n3} to {name2}. How many {item} do they have left?",
            "completion": "{n1} + {n2} = {s1}; {s1} - {n3} = {s2}"
        },
        {
            "type": "2-step",
            "prompt": "A bakery has {n1} cakes. They bake {n2} more in the morning and sell {n3} in the afternoon. How many cakes are left at the end of the day?",
            "completion": "{n1} + {n2} = {s1}; {s1} - {n3} = {s2}"
        },
        {
            "type": "3-step",
            "prompt": "{name1} has {n1} {item}. {name2} gives them {n2} {item}, and they find another {n3}. Then, they lose {n4}. How many {item} remain?",
            "completion": "{n1} + {n2} = {s1}; {s1} + {n3} = {s2}; {s2} - {n4} = {s3}"
        }
    ]
    
    template = random.choice(templates)
    
    name1 = fake.first_name()
    name2 = fake.first_name()
    item = random.choice(['apples', 'books', 'coins', 'marbles', 'stickers'])
    
    if template["type"] == "2-step":
        n1 = random.randint(20, 50)
        n2 = random.randint(5, 25)
        n3 = random.randint(1, n1 + n2 - 1)
        s1 = n1 + n2
        s2 = s1 - n3
        
        prompt = template["prompt"].format(name1=name1, name2=name2, item=item, n1=n1, n2=n2, n3=n3)
        completion = template["completion"].format(n1=n1, n2=n2, s1=s1, n3=n3, s2=s2)
        
    elif template["type"] == "3-step":
        n1 = random.randint(20, 50)
        n2 = random.randint(5, 25)
        n3 = random.randint(1, 15)
        intermediate_sum = n1 + n2 + n3
        n4 = random.randint(1, intermediate_sum - 1)
        s1 = n1 + n2
        s2 = s1 + n3
        s3 = s2 - n4
        
        prompt = template["prompt"].format(name1=name1, name2=name2, item=item, n1=n1, n2=n2, n3=n3, n4=n4)
        completion = template["completion"].format(n1=n1, n2=n2, s1=s1, n3=n3, s2=s2, n4=n4, s3=s3)

    return {"prompt": prompt, "completion": completion}

# --- Main script ---
if __name__ == "__main__":
    print("Generating multi-step math word problem dataset...")
    
    all_data = [create_multistep_problem() for _ in range(1000)]
    
    random.shuffle(all_data)
    
    train_data = all_data[:900]
    val_data = all_data[900:]
    
    # Save to JSONL files
    with open('train_multistep.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
            
    with open('validation_multistep.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Dataset generated successfully!")
    print(f"Created train_multistep.jsonl ({len(train_data)}) and validation_multistep.jsonl ({len(val_data)}).")
    print("\nExample from training data:")
    print(train_data[0])
