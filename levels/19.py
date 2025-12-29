allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = 10

def main():
    while True:
        a = int(input())
        while a != 0:
            print(a)
            if a > 0:
                a = a - 1
            else:
                a = a + 1
        # include zero in the output
        print(a)



if __name__ == "__main__":
    main()
