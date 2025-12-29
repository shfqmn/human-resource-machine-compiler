allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = 10

def main():
    while True:
        a = int(input())
        b = int(input())

        if b == 0:
            print(b)
        elif a == 0:
            print(a)
        else:
            while a >= b:
                a = a - b
            print(a)



if __name__ == "__main__":
    main()
