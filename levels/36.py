allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles: list = [
    None, None, None, None, None, 
    None, None, None, None, None, 
    None, None, None, None, None, 
    None, None, None, None, None, 
    None, None,None, 0, 10,
    ]

def main():
    while True:
        # Read first word
        p_write = 0
        
        ch = input()
        while ch != "0":
            tiles[p_write] = ch
            p_write = p_write + 1
            ch = input()
        
        p1_end = p_write
        
        # Read second word
        ch = input()
        while ch != "0":
            tiles[p_write] = ch
            p_write = p_write + 1
            ch = input()
            
        # p_write is now p2_end
        
        # Compare
        p1_curr = 0
        p2_curr = p1_end
        
        pick_first = False
        
        while True:
            if p1_curr == p1_end:
                # Word 1 ended. Word 1 is prefix of Word 2.
                pick_first = True
                break
                
            if p2_curr == p_write: # p_write is p2_end
                # Word 2 ended. Word 2 is prefix of Word 1.
                pick_first = False
                break
            
            ch = tiles[p1_curr] # Reuse ch as val1
            val2 = tiles[p2_curr]
            
            if ch != val2:
                if ch < val2:
                    pick_first = True
                else:
                    pick_first = False
                break
            
            p1_curr = p1_curr + 1
            p2_curr = p2_curr + 1
            
        # Output
        # Reuse p1_curr as out_curr
        # Reuse p2_curr as out_end
        
        if pick_first:
            p1_curr = 0
            p2_curr = p1_end
        else:
            p1_curr = p1_end
            p2_curr = p_write
            
        while p1_curr < p2_curr:
            print(tiles[p1_curr])
            p1_curr = p1_curr + 1

if __name__ == "__main__":
    main()
