allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None,
    None, None, None, None,
    None, None, None, None,
    None, None, 0, 4,
    ]

def main():
    while True:
        n = int(input())

        if n < 4: # row 0
            print(n) # x
            print(0) # y
        elif n < 8: # row 1
            print(n - 4) # x
            print(1)     # y
        elif n < 12: # row 2
            print(n - 8) # x
            print(2)     # y
        elif n < 16: # row 3
            print(n - 12) # x
            print(3)      # y


if __name__ == "__main__":
    main()
