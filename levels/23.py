allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None,
    None, None, None, None,
    ]

def main():
    while True:
        smallest = int(input())
        while True:
            a = int(input())
            if a == 0:
                break
            if a < smallest:
                smallest = a
        print(smallest)

if __name__ == "__main__":
    main()
