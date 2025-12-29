allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    "B", "A", "X", "B", "C",
    "X", "A", "B", "A", "X",
    "C", "B", "A", "B",  0 ,
    None, None, None, None, None
    ]
modifiable_tiles_idx = [15,16,17,18,19]

def main():
    a = 4
    b = 5
    c = 2
    x = 3
    
    while True:
        ch = input()
        if ch == "A":
            print(a)
        elif ch == "B":
            print(b)
        elif ch == "C":
            print(c)
        else:
            print(x)

if __name__ == "__main__":
    main()
