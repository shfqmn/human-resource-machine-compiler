allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "JUMP"]
# Use a literal `tiles` value so the HRM compiler can parse it.
tiles = 3

def main():
    while True:
        a = input()
        b = input()
        print(b)
        print(a)


if __name__ == "__main__":
    main()
