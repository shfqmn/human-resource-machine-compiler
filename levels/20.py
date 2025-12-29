allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None, None, 
    None, None, None, None, 0
    ]

def main():
    while True:
        a = int(input())
        b = int(input())
        if a == 0 or b == 0:
            print(0)
        else:
            c = a * b
            print(c)



if __name__ == "__main__":
    main()
