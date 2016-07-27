
from matplotlib.colors import ListedColormap
from numpy import nan, inf

# Used to reconstruct the colormap in viscm
parameters = {'xp': [24.397622053872112, 22.329064134250359, -3.5279098610215271, -7.3202660469947318, -3.5279098610215271],
              'yp': [10.224586288416106, 29.531126871552431, 21.60165484633572, 1.6055949566588197, -1.4972419227738101],
              'min_Jp': 18.730158730158728,
              'max_Jp': 99.6190476190476}

cm_data = [[  3.19188006e-01,  5.08411011e-04,  4.27847166e-02],
           [  3.23452429e-01,  2.94004376e-03,  4.29498405e-02],
           [  3.27697959e-01,  5.48550076e-03,  4.30937800e-02],
           [  3.31925014e-01,  8.14729136e-03,  4.32108710e-02],
           [  3.36134026e-01,  1.09282090e-02,  4.32932080e-02],
           [  3.40324261e-01,  1.38309580e-02,  4.33544174e-02],
           [  3.44496109e-01,  1.68585499e-02,  4.33863092e-02],
           [  3.48649544e-01,  2.00141344e-02,  4.33873393e-02],
           [  3.52784075e-01,  2.33004930e-02,  4.33674127e-02],
           [  3.56899901e-01,  2.67211674e-02,  4.33188700e-02],
           [  3.60996892e-01,  3.02796868e-02,  4.32405671e-02],
           [  3.65074618e-01,  3.39789376e-02,  4.31416825e-02],
           [  3.69133039e-01,  3.78227502e-02,  4.30181093e-02],
           [  3.73172062e-01,  4.17796063e-02,  4.28631975e-02],
           [  3.77191242e-01,  4.56665666e-02,  4.26883879e-02],
           [  3.81190385e-01,  4.94889072e-02,  4.24937807e-02],
           [  3.85169297e-01,  5.32578705e-02,  4.22682848e-02],
           [  3.89127602e-01,  5.69807657e-02,  4.20217552e-02],
           [  3.93065023e-01,  6.06644181e-02,  4.17567112e-02],
           [  3.96981266e-01,  6.43150594e-02,  4.14725462e-02],
           [  4.00875841e-01,  6.79394110e-02,  4.11600480e-02],
           [  4.04748512e-01,  7.15406015e-02,  4.08309528e-02],
           [  4.08598937e-01,  7.51227174e-02,  4.04858381e-02],
           [  4.12426752e-01,  7.86893890e-02,  4.01206455e-02],
           [  4.16231263e-01,  8.22452517e-02,  3.97338038e-02],
           [  4.20012268e-01,  8.57920330e-02,  3.93348179e-02],
           [  4.23769406e-01,  8.93321242e-02,  3.89262925e-02],
           [  4.27502256e-01,  9.28678889e-02,  3.85095772e-02],
           [  4.31210387e-01,  9.64014725e-02,  3.80861373e-02],
           [  4.34892901e-01,  9.99361677e-02,  3.76509851e-02],
           [  4.38549738e-01,  1.03472432e-01,  3.72127744e-02],
           [  4.42180470e-01,  1.07011798e-01,  3.67737224e-02],
           [  4.45784643e-01,  1.10555736e-01,  3.63357812e-02],
           [  4.49361798e-01,  1.14105584e-01,  3.59010342e-02],
           [  4.52911477e-01,  1.17662562e-01,  3.54716989e-02],
           [  4.56432847e-01,  1.21228614e-01,  3.50467949e-02],
           [  4.59925736e-01,  1.24804054e-01,  3.46317940e-02],
           [  4.63389808e-01,  1.28389545e-01,  3.42302550e-02],
           [  4.66824637e-01,  1.31985895e-01,  3.38449160e-02],
           [  4.70229810e-01,  1.35593819e-01,  3.34786526e-02],
           [  4.73604927e-01,  1.39213951e-01,  3.31344783e-02],
           [  4.76949608e-01,  1.42846843e-01,  3.28155440e-02],
           [  4.80263491e-01,  1.46492970e-01,  3.25251366e-02],
           [  4.83546237e-01,  1.50152732e-01,  3.22666779e-02],
           [  4.86797340e-01,  1.53826781e-01,  3.20427670e-02],
           [  4.90016620e-01,  1.57515177e-01,  3.18577613e-02],
           [  4.93203933e-01,  1.61217925e-01,  3.17160407e-02],
           [  4.96359052e-01,  1.64935164e-01,  3.16215131e-02],
           [  4.99481779e-01,  1.68666973e-01,  3.15782102e-02],
           [  5.02571949e-01,  1.72413379e-01,  3.15902848e-02],
           [  5.05629430e-01,  1.76174354e-01,  3.16620080e-02],
           [  5.08654123e-01,  1.79949823e-01,  3.17977662e-02],
           [  5.11645963e-01,  1.83739667e-01,  3.20020585e-02],
           [  5.14604920e-01,  1.87543723e-01,  3.22794932e-02],
           [  5.17530997e-01,  1.91361790e-01,  3.26347858e-02],
           [  5.20424232e-01,  1.95193632e-01,  3.30727557e-02],
           [  5.23284694e-01,  1.99038983e-01,  3.35983244e-02],
           [  5.26112485e-01,  2.02897548e-01,  3.42165124e-02],
           [  5.28907738e-01,  2.06769007e-01,  3.49324380e-02],
           [  5.31670618e-01,  2.10653019e-01,  3.57513149e-02],
           [  5.34401314e-01,  2.14549225e-01,  3.66784508e-02],
           [  5.37100046e-01,  2.18457250e-01,  3.77192462e-02],
           [  5.39767058e-01,  2.22376711e-01,  3.88791931e-02],
           [  5.42402617e-01,  2.26307210e-01,  4.01638742e-02],
           [  5.45007013e-01,  2.30248347e-01,  4.15510718e-02],
           [  5.47580556e-01,  2.34199717e-01,  4.30359691e-02],
           [  5.50123575e-01,  2.38160913e-01,  4.46216703e-02],
           [  5.52636413e-01,  2.42131527e-01,  4.63067669e-02],
           [  5.55119432e-01,  2.46111156e-01,  4.80895724e-02],
           [  5.57573004e-01,  2.50099400e-01,  4.99681664e-02],
           [  5.59997514e-01,  2.54095864e-01,  5.19404378e-02],
           [  5.62393356e-01,  2.58100161e-01,  5.40041260e-02],
           [  5.64760934e-01,  2.62111911e-01,  5.61568596e-02],
           [  5.67100658e-01,  2.66130744e-01,  5.83961918e-02],
           [  5.69412966e-01,  2.70156278e-01,  6.07196404e-02],
           [  5.71698290e-01,  2.74188154e-01,  6.31246983e-02],
           [  5.73957035e-01,  2.78226051e-01,  6.56088561e-02],
           [  5.76189628e-01,  2.82269644e-01,  6.81696432e-02],
           [  5.78396498e-01,  2.86318616e-01,  7.08046350e-02],
           [  5.80578073e-01,  2.90372663e-01,  7.35114667e-02],
           [  5.82734784e-01,  2.94431495e-01,  7.62878435e-02],
           [  5.84867059e-01,  2.98494830e-01,  7.91315492e-02],
           [  5.86975325e-01,  3.02562403e-01,  8.20404518e-02],
           [  5.89060008e-01,  3.06633957e-01,  8.50125072e-02],
           [  5.91121531e-01,  3.10709248e-01,  8.80457618e-02],
           [  5.93160315e-01,  3.14788043e-01,  9.11383525e-02],
           [  5.95176778e-01,  3.18870120e-01,  9.42885072e-02],
           [  5.97171334e-01,  3.22955269e-01,  9.74945428e-02],
           [  5.99144397e-01,  3.27043288e-01,  1.00754863e-01],
           [  6.01096374e-01,  3.31133987e-01,  1.04067958e-01],
           [  6.03027671e-01,  3.35227184e-01,  1.07432397e-01],
           [  6.04938690e-01,  3.39322707e-01,  1.10846829e-01],
           [  6.06829831e-01,  3.43420394e-01,  1.14309978e-01],
           [  6.08701490e-01,  3.47520088e-01,  1.17820639e-01],
           [  6.10554058e-01,  3.51621644e-01,  1.21377673e-01],
           [  6.12387926e-01,  3.55724921e-01,  1.24980008e-01],
           [  6.14203480e-01,  3.59829787e-01,  1.28626629e-01],
           [  6.16001408e-01,  3.63935903e-01,  1.32316392e-01],
           [  6.17781818e-01,  3.68043347e-01,  1.36048549e-01],
           [  6.19545070e-01,  3.72152022e-01,  1.39822260e-01],
           [  6.21291540e-01,  3.76261820e-01,  1.43636721e-01],
           [  6.23021600e-01,  3.80372642e-01,  1.47491171e-01],
           [  6.24735623e-01,  3.84484392e-01,  1.51384889e-01],
           [  6.26433979e-01,  3.88596978e-01,  1.55317195e-01],
           [  6.28117036e-01,  3.92710314e-01,  1.59287442e-01],
           [  6.29785163e-01,  3.96824318e-01,  1.63295020e-01],
           [  6.31438727e-01,  4.00938911e-01,  1.67339345e-01],
           [  6.33078094e-01,  4.05054018e-01,  1.71419867e-01],
           [  6.34703629e-01,  4.09169567e-01,  1.75536060e-01],
           [  6.36315954e-01,  4.13285331e-01,  1.79687212e-01],
           [  6.37915317e-01,  4.17401322e-01,  1.83872936e-01],
           [  6.39501936e-01,  4.21517570e-01,  1.88092897e-01],
           [  6.41076174e-01,  4.25634014e-01,  1.92346660e-01],
           [  6.42638397e-01,  4.29750599e-01,  1.96633813e-01],
           [  6.44188969e-01,  4.33867270e-01,  2.00953958e-01],
           [  6.45728257e-01,  4.37983973e-01,  2.05306713e-01],
           [  6.47256627e-01,  4.42100659e-01,  2.09691714e-01],
           [  6.48774449e-01,  4.46217278e-01,  2.14108606e-01],
           [  6.50282093e-01,  4.50333782e-01,  2.18557051e-01],
           [  6.51780194e-01,  4.54449979e-01,  2.23036476e-01],
           [  6.53268943e-01,  4.58565931e-01,  2.27546728e-01],
           [  6.54748608e-01,  4.62681653e-01,  2.32087599e-01],
           [  6.56219567e-01,  4.66797103e-01,  2.36658794e-01],
           [  6.57682196e-01,  4.70912243e-01,  2.41260026e-01],
           [  6.59136877e-01,  4.75027032e-01,  2.45891017e-01],
           [  6.60583993e-01,  4.79141432e-01,  2.50551496e-01],
           [  6.62023931e-01,  4.83255406e-01,  2.55241197e-01],
           [  6.63457080e-01,  4.87368916e-01,  2.59959863e-01],
           [  6.64884050e-01,  4.91481816e-01,  2.64707020e-01],
           [  6.66305021e-01,  4.95594182e-01,  2.69482635e-01],
           [  6.67720349e-01,  4.99706002e-01,  2.74286507e-01],
           [  6.69130435e-01,  5.03817240e-01,  2.79118400e-01],
           [  6.70535682e-01,  5.07927863e-01,  2.83978083e-01],
           [  6.71936499e-01,  5.12037840e-01,  2.88865326e-01],
           [  6.73333295e-01,  5.16147137e-01,  2.93779904e-01],
           [  6.74726487e-01,  5.20255723e-01,  2.98721596e-01],
           [  6.76116551e-01,  5.24363538e-01,  3.03690119e-01],
           [  6.77503877e-01,  5.28470568e-01,  3.08685290e-01],
           [  6.78888821e-01,  5.32576816e-01,  3.13706971e-01],
           [  6.80271812e-01,  5.36682249e-01,  3.18754950e-01],
           [  6.81653287e-01,  5.40786839e-01,  3.23829018e-01],
           [  6.83033683e-01,  5.44890554e-01,  3.28928966e-01],
           [  6.84413445e-01,  5.48993365e-01,  3.34054587e-01],
           [  6.85793019e-01,  5.53095241e-01,  3.39205675e-01],
           [  6.87172842e-01,  5.57196161e-01,  3.44382044e-01],
           [  6.88553325e-01,  5.61296115e-01,  3.49583542e-01],
           [  6.89934944e-01,  5.65395067e-01,  3.54809950e-01],
           [  6.91318165e-01,  5.69492986e-01,  3.60061066e-01],
           [  6.92703460e-01,  5.73589844e-01,  3.65336688e-01],
           [  6.94091305e-01,  5.77685610e-01,  3.70636614e-01],
           [  6.95482183e-01,  5.81780256e-01,  3.75960642e-01],
           [  6.96876579e-01,  5.85873751e-01,  3.81308568e-01],
           [  6.98274958e-01,  5.89966078e-01,  3.86680224e-01],
           [  6.99677633e-01,  5.94057281e-01,  3.92075624e-01],
           [  7.01085271e-01,  5.98147261e-01,  3.97494373e-01],
           [  7.02498381e-01,  6.02235988e-01,  4.02936266e-01],
           [  7.03917477e-01,  6.06323433e-01,  4.08401097e-01],
           [  7.05343079e-01,  6.10409565e-01,  4.13888660e-01],
           [  7.06775712e-01,  6.14494353e-01,  4.19398747e-01],
           [  7.08215906e-01,  6.18577766e-01,  4.24931150e-01],
           [  7.09664198e-01,  6.22659774e-01,  4.30485657e-01],
           [  7.11120918e-01,  6.26740423e-01,  4.36062323e-01],
           [  7.12586632e-01,  6.30819674e-01,  4.41660917e-01],
           [  7.14062051e-01,  6.34897436e-01,  4.47281028e-01],
           [  7.15547738e-01,  6.38973677e-01,  4.52922438e-01],
           [  7.17044260e-01,  6.43048362e-01,  4.58584925e-01],
           [  7.18552191e-01,  6.47121460e-01,  4.64268267e-01],
           [  7.20072113e-01,  6.51192935e-01,  4.69972235e-01],
           [  7.21604612e-01,  6.55262754e-01,  4.75696601e-01],
           [  7.23150282e-01,  6.59330882e-01,  4.81441132e-01],
           [  7.24709638e-01,  6.63397313e-01,  4.87205706e-01],
           [  7.26282965e-01,  6.67462115e-01,  4.92990519e-01],
           [  7.27871260e-01,  6.71525124e-01,  4.98794821e-01],
           [  7.29475145e-01,  6.75586301e-01,  5.04618361e-01],
           [  7.31095251e-01,  6.79645609e-01,  5.10460886e-01],
           [  7.32732213e-01,  6.83703007e-01,  5.16322138e-01],
           [  7.34386675e-01,  6.87758456e-01,  5.22201851e-01],
           [  7.36059289e-01,  6.91811916e-01,  5.28099756e-01],
           [  7.37750713e-01,  6.95863345e-01,  5.34015578e-01],
           [  7.39461613e-01,  6.99912703e-01,  5.39949035e-01],
           [  7.41192662e-01,  7.03959945e-01,  5.45899838e-01],
           [  7.42944541e-01,  7.08005028e-01,  5.51867692e-01],
           [  7.44717602e-01,  7.12048006e-01,  5.57852798e-01],
           [  7.46512751e-01,  7.16088771e-01,  5.63854543e-01],
           [  7.48330822e-01,  7.20127236e-01,  5.69872417e-01],
           [  7.50172531e-01,  7.24163354e-01,  5.75906084e-01],
           [  7.52038600e-01,  7.28197076e-01,  5.81955204e-01],
           [  7.53929761e-01,  7.32228351e-01,  5.88019423e-01],
           [  7.55846753e-01,  7.36257131e-01,  5.94098377e-01],
           [  7.57790321e-01,  7.40283361e-01,  6.00191688e-01],
           [  7.59761221e-01,  7.44306992e-01,  6.06298969e-01],
           [  7.61760216e-01,  7.48327968e-01,  6.12419817e-01],
           [  7.63788076e-01,  7.52346237e-01,  6.18553814e-01],
           [  7.65845580e-01,  7.56361743e-01,  6.24700530e-01],
           [  7.67933513e-01,  7.60374432e-01,  6.30859514e-01],
           [  7.70052669e-01,  7.64384247e-01,  6.37030303e-01],
           [  7.72203849e-01,  7.68391133e-01,  6.43212411e-01],
           [  7.74387858e-01,  7.72395033e-01,  6.49405338e-01],
           [  7.76605512e-01,  7.76395891e-01,  6.55608559e-01],
           [  7.78857629e-01,  7.80393651e-01,  6.61821529e-01],
           [  7.81145034e-01,  7.84388257e-01,  6.68043683e-01],
           [  7.83468558e-01,  7.88379654e-01,  6.74274427e-01],
           [  7.85829033e-01,  7.92367789e-01,  6.80513147e-01],
           [  7.88227297e-01,  7.96352608e-01,  6.86759199e-01],
           [  7.90664188e-01,  8.00334062e-01,  6.93011914e-01],
           [  7.93140546e-01,  8.04312100e-01,  6.99270593e-01],
           [  7.95657211e-01,  8.08286679e-01,  7.05534505e-01],
           [  7.98215019e-01,  8.12257756e-01,  7.11802892e-01],
           [  8.00814805e-01,  8.16225293e-01,  7.18074960e-01],
           [  8.03457395e-01,  8.20189256e-01,  7.24349883e-01],
           [  8.06143611e-01,  8.24149618e-01,  7.30626801e-01],
           [  8.08874262e-01,  8.28106357e-01,  7.36904818e-01],
           [  8.11650145e-01,  8.32059461e-01,  7.43183002e-01],
           [  8.14472043e-01,  8.36008923e-01,  7.49460385e-01],
           [  8.17340716e-01,  8.39954747e-01,  7.55735965e-01],
           [  8.20256906e-01,  8.43896948e-01,  7.62008700e-01],
           [  8.23221044e-01,  8.47835583e-01,  7.68278333e-01],
           [  8.26233704e-01,  8.51770691e-01,  7.74544189e-01],
           [  8.29295924e-01,  8.55702279e-01,  7.80804058e-01],
           [  8.32408327e-01,  8.59630412e-01,  7.87056769e-01],
           [  8.35571491e-01,  8.63555173e-01,  7.93301118e-01],
           [  8.38785950e-01,  8.67476662e-01,  7.99535878e-01],
           [  8.42052188e-01,  8.71395001e-01,  8.05759795e-01],
           [  8.45370193e-01,  8.75310323e-01,  8.11973366e-01],
           [  8.48740412e-01,  8.79222758e-01,  8.18175322e-01],
           [  8.52163579e-01,  8.83132479e-01,  8.24362799e-01],
           [  8.55639953e-01,  8.87039686e-01,  8.30534495e-01],
           [  8.59169726e-01,  8.90944601e-01,  8.36689115e-01],
           [  8.62752326e-01,  8.94847329e-01,  8.42829467e-01],
           [  8.66388535e-01,  8.98748189e-01,  8.48950924e-01],
           [  8.70078463e-01,  9.02647473e-01,  8.55051709e-01],
           [  8.73821857e-01,  9.06545384e-01,  8.61132327e-01],
           [  8.77618473e-01,  9.10442054e-01,  8.67193908e-01],
           [  8.81468712e-01,  9.14337976e-01,  8.73231456e-01],
           [  8.85372253e-01,  9.18233284e-01,  8.79246333e-01],
           [  8.89328979e-01,  9.22128095e-01,  8.85239267e-01],
           [  8.93339074e-01,  9.26022993e-01,  8.91205135e-01],
           [  8.97402366e-01,  9.29917617e-01,  8.97149605e-01],
           [  9.01519093e-01,  9.33812678e-01,  9.03066039e-01],
           [  9.05689518e-01,  9.37707793e-01,  9.08958623e-01],
           [  9.09913938e-01,  9.41603448e-01,  9.14822730e-01],
           [  9.14193252e-01,  9.45499026e-01,  9.20662416e-01],
           [  9.18527956e-01,  9.49395047e-01,  9.26471928e-01],
           [  9.22919852e-01,  9.53290490e-01,  9.32255874e-01],
           [  9.27370222e-01,  9.57185424e-01,  9.38009990e-01],
           [  9.31881046e-01,  9.61079371e-01,  9.43732754e-01],
           [  9.36455353e-01,  9.64971274e-01,  9.49424410e-01],
           [  9.41096680e-01,  9.68860078e-01,  9.55083193e-01],
           [  9.45808721e-01,  9.72744936e-01,  9.60704680e-01],
           [  9.50596398e-01,  9.76624444e-01,  9.66285224e-01],
           [  9.55465459e-01,  9.80497024e-01,  9.71819696e-01],
           [  9.60422407e-01,  9.84360987e-01,  9.77301129e-01],
           [  9.65474320e-01,  9.88214632e-01,  9.82720362e-01],
           [  9.70629792e-01,  9.92055757e-01,  9.88067257e-01],
           [  9.75895168e-01,  9.95883564e-01,  9.93326172e-01],
           [  9.81275361e-01,  9.99697973e-01,  9.98479623e-01]]

test_cm = ListedColormap(cm_data, name=__file__)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from viscm import viscm
        viscm(test_cm)
    except ImportError:
        print("viscm not found, falling back on simple display")
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',
                   cmap=test_cm)
    plt.show()
