# X22 (DRAFT - pending X21 verdict): early-window dynamics + cross-stream interactions.
# Buckets [0,10),[10,30),[30,60),[60,180),[180,300),[300,600) where useful.
# Families:
#  a) book slope early: (a2-a1)/mid, (b1-b2)/mid means in [0,10),[10,30),[30,60)
#  b) window deltas: signed tx vol (0-10) minus (10-30); L1 imb (0-10) minus (30-60); spread (0-10) minus (300-600)
#  c) cross-stream: signed tx vol * L1 imb co-movement per bucket: mean(signvol_i * imb_i) over market/tx-joined bars is
#     not directly streamable; approximation: corr proxy via sum(svol)*simb products per bucket -> cov(svol, imb) per sample
#  d) cancel-to-new ratio per side per early bucket
# NOTE: implement only after X21 verdict decides whether early-window family has legs.
