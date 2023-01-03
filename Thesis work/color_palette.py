"darkgoldenrod"
"gray"
"gray"
"tab:brown"
"sienna" #
"sandybrown" #
"peachpuff"
"orange"
"green"
"moccasin" #

"dimgray"
"darkorange"




'''
# PLOT ---
x_ = [[P[0][i][j], P[0][i][j + 1]]
      for i in range(len(P[0]))
      for j in range(len(P[0][0]) - 1)]

y_ = [[P[1][i][j], P[1][i][j + 1]]
      for i in range(len(P[0]))
      for j in range(len(P[0][0]) - 1)]

z_ = [[P[2][i][j], P[2][i][j + 1]]
      for i in range(len(P[0]))
      for j in range(len(P[0][0]) - 1)]
################
        # PLOT
        x_ = []
        y_ = []
        z_ = []

        for i in range(len(P[0])):
            for j in range(len(P[0][0]) - 1):
                tmp_x_ = [P[0][i][j], P[0][i][j + 1]]
                tmp_y_ = [P[1][i][j], P[1][i][j + 1]]
                tmp_z_ = [P[2][i][j], P[2][i][j + 1]]
                x_.append(tmp_x_)
                y_.append(tmp_y_)
                z_.append(tmp_z_)

'''