 # reba score calculation - groupA
def groupA(neck_score, trunk_score, r_leg_score):
    if neck_score == 1:
        if r_leg_score == 1:
            if trunk_score == 1:
                return 1
            elif trunk_score == 2:
                return 2
            elif trunk_score == 3:
                return 2
            elif trunk_score == 4:
                return 3
            elif trunk_score == 5:
                return 4    
        elif r_leg_score == 2:
            if trunk_score == 1:
                return 2
            elif trunk_score == 2:
                return 3
            elif trunk_score == 3:
                return 4
            elif trunk_score == 4:
                return 5
            elif trunk_score == 5:  
                return 6
        elif r_leg_score == 3:
            if trunk_score == 1:
                return 3
            elif trunk_score == 2:
                return 4
            elif trunk_score == 3:
                return 5
            elif trunk_score == 4:
                return 6
            elif trunk_score == 5:
                return 7
        elif r_leg_score == 4:
            if trunk_score == 1:
                return 4
            elif trunk_score == 2:
                return 5
            elif trunk_score == 3:
                return 6
            elif trunk_score == 4:
                return 7
            elif trunk_score == 5:
                return 8          
    elif neck_score == 2:
        if r_leg_score == 1:
            if trunk_score == 1:
                return 1
            elif trunk_score == 2:
                return 3
            elif trunk_score == 3:
                return 4
            elif trunk_score == 4:
                return 5
            elif trunk_score == 5:
                return 6    
        elif r_leg_score == 2:
            if trunk_score == 1:
                return 2
            elif trunk_score == 2:
                return 4
            elif trunk_score == 3:
                return 5
            elif trunk_score == 4:
                return 6
            elif trunk_score == 5:  
                return 7
        elif r_leg_score == 3:
            if trunk_score == 1:
                return 3
            elif trunk_score == 2:
                return 5
            elif trunk_score == 3:
                return 6
            elif trunk_score == 4:
                return 7
            elif trunk_score == 5:
                return 8
        elif r_leg_score == 4:
            if trunk_score == 1:
                return 4
            elif trunk_score == 2:
                return 6
            elif trunk_score == 3:
                return 7
            elif trunk_score == 4:
                return 8
            elif trunk_score == 5:
                return 9          
    elif neck_score == 3:
        if r_leg_score == 1:
            if trunk_score == 1:
                return 3
            elif trunk_score == 2:
                return 4
            elif trunk_score == 3:
                return 5
            elif trunk_score == 4:
                return 6
            elif trunk_score == 5:
                return 7    
        elif r_leg_score == 2:
            if trunk_score == 1:
                return 3
            elif trunk_score == 2:
                return 5
            elif trunk_score == 3:
                return 6
            elif trunk_score == 4:
                return 7
            elif trunk_score == 5:  
                return 8
        elif r_leg_score == 3:
            if trunk_score == 1:
                return 5
            elif trunk_score == 2:
                return 6
            elif trunk_score == 3:
                return 7
            elif trunk_score == 4:
                return 8
            elif trunk_score == 5:
                return 9
        elif r_leg_score == 4:
            if trunk_score == 1:
                return 6
            elif trunk_score == 2:
                return 7
            elif trunk_score == 3:
                return 8
            elif trunk_score == 4:
                return 9
            elif trunk_score == 5:
                return 9            


 # reba score calculation - groupB
def groupB(r_lower_arm_score, r_wrist_score, r_upper_arm_score):
    if r_lower_arm_score == 1:
        if r_wrist_score == 1:
            if r_upper_arm_score == 1:
                return 1
            elif r_upper_arm_score == 2:
                return 1
            elif r_upper_arm_score == 3:
                return 3
            elif r_upper_arm_score == 4:
                return 4
            elif r_upper_arm_score == 5:
                return 6 
            elif r_upper_arm_score == 6:
                return 7    
        elif r_wrist_score == 2:
            if r_upper_arm_score == 1:
                return 2
            elif r_upper_arm_score == 2:
                return 2
            elif r_upper_arm_score == 3:
                return 4
            elif r_upper_arm_score == 4:
                return 5
            elif r_upper_arm_score == 5:
                return 7 
            elif r_upper_arm_score == 6:
                return 8
        elif r_wrist_score == 3:
            if r_upper_arm_score == 1:
                return 2
            elif r_upper_arm_score == 2:
                return 3
            elif r_upper_arm_score == 3:
                return 5
            elif r_upper_arm_score == 4:
                return 5
            elif r_upper_arm_score == 5:
                return 8 
            elif r_upper_arm_score == 6:
                return 8
    elif r_lower_arm_score == 2:
        if r_wrist_score == 1:
            if r_upper_arm_score == 1:
                return 1
            elif r_upper_arm_score == 2:
                return 2
            elif r_upper_arm_score == 3:
                return 4
            elif r_upper_arm_score == 4:
                return 5
            elif r_upper_arm_score == 5:
                return 7 
            elif r_upper_arm_score == 6:
                return 8    
        elif r_wrist_score == 2:
            if r_upper_arm_score == 1:
                return 2
            elif r_upper_arm_score == 2:
                return 3
            elif r_upper_arm_score == 3:
                return 5
            elif r_upper_arm_score == 4:
                return 6
            elif r_upper_arm_score == 5:
                return 8 
            elif r_upper_arm_score == 6:
                return 9
        elif r_wrist_score == 3:
            if r_upper_arm_score == 1:
                return 3
            elif r_upper_arm_score == 2:
                return 4
            elif r_upper_arm_score == 3:
                return 5
            elif r_upper_arm_score == 4:
                return 7
            elif r_upper_arm_score == 5:
                return 8 
            elif r_upper_arm_score == 6:
                return 9


def groupC(groupA, groupB):
    i = groupA -1
    j = groupB -1
    arr = [[1,1,2,3,4,6,7,8,9],
           [1,2,3,4,4,6,7,8,9],
           [1,2,3,4,4,6,7,8,9],
           [2,3,3,4,5,7,8,9,10],
           [3,4,4,5,6,8,9,10,10],
           [3,4,5,6,7,8,9,10,10],
           [4,5,6,7,8,9,9,10,11],
           [5,6,7,8,8,9,10,10,11],
           [6,6,7,8,9,10,10,10,11]]
    return arr[j][i]