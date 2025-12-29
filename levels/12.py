allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "JUMP"]
tiles = 5

def main():
    while True:
        a = int(input())
        five_a = a + a + a + a + a
        four_five_a = five_a + five_a + five_a + five_a
        two_four_five_a = four_five_a + four_five_a
        print(two_four_five_a)

if __name__ == "__main__":
    main()
