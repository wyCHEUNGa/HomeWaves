import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
# import seaborn as sns
import datetime
import math

import streamlit as st # web development

###Credits Sakari Cajanus 1 Mar 2020 on https://stackoverflow.com/questions/49827096/generating-a-plotly-heat-map-from-a-pandas-pivot-table
def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

###//2023
def update_logs091023(low, high, fd, filename):
    with open(filename, 'a') as f:
        if fd == 1:
            f.write(str(low) + "\t" + str(high-1) + "\t" + str(fd) + "\t" + "No file\n")
        else:
            f.write(str(low) + "\t" + str(high-1) + "\t" + str(fd) + "\t" + "See ---.txt\n")

###//2023
def peak_detected(b_l, low, high, fd):
    temp=[]

    if fd == 1:
        temp[:40] = np.zeros(0)

    else:
    #{'1':low,'2':high-1,'3':fd,'4':"add results of sliding window"}

        for h in range(0,40):
            with open("./data/b_l_range"+str(low)+"_to_"+str(high-1)+".txt"    , "a") as branch_file:

                for w in range(0,141):
                    #TESTER w=5; h=20

                    if h==0:
                        temp.append(0)

                    elif h==1:
                        temp.append(0)

                    elif (1<h<38):
                        if w <= 3: 
                            temp.append(0)
                        elif (3< w <137):
                            ### Starting Peak Detection. Creation of 9x5 sliding window. 
                            th_motion = (
                            b_l[w-4][h-2]+
                            b_l[w-3][h-2]+
                            b_l[w-4][h-1]+
                            b_l[w-3][h-1]+
                            b_l[w-4][h-0]+
                            b_l[w-3][h-0]+
                            b_l[w-4][h+1]+
                            b_l[w-3][h+1]+
                            b_l[w-4][h+2]+
                            b_l[w-3][h+2]+

                            b_l[w-2][h-2]+
                            b_l[w-1][h-2]+
                            b_l[w-0][h-2]+
                            b_l[w+1][h-2]+
                            b_l[w+2][h-2]+

                            b_l[w+4][h-2]+
                            b_l[w+3][h-2]+
                            b_l[w+4][h-1]+
                            b_l[w+3][h-1]+
                            b_l[w+4][h-0]+
                            b_l[w+3][h-0]+
                            b_l[w+4][h+1]+
                            b_l[w+3][h+1]+
                            b_l[w+4][h+2]+
                            b_l[w+3][h+2]+

                            b_l[w-2][h+2]+
                            b_l[w-1][h+2]+
                            b_l[w-0][h+2]+
                            b_l[w+1][h+2]+
                            b_l[w+2][h+2])/30

                            #print("b_l["    ,w,   "]["   ,h,   "] is ",
                            #      np.round(b_l[w][h],6), " \t 1.5*noise threshold is ", np.round(1.5*th_motion,6))

                            branch_file.write("b_l["  +  str(w)  +  "]["  +  str(h)  +  "] is "  +
                                str(np.round(b_l[w][h],6))  +  " \t 1.5*noise threshold is "  +  str(np.round(1.5*th_motion,6))+"\n")
                            if b_l[w][h]>=1.5*th_motion:
                                branch_file.write("Yes human detected.\n")
                                temp.append(1)
                            else:
                                branch_file.write("Move to next.\n")
                                temp.append(0)
                        else: #w gt or equal to 137
                            temp.append(0)

                    elif h==38:
                        temp.append(0)

                    elif h==39:
                        temp.append(0)
        
    update_logs091023(low, high, fd, "./data/logs"+DDMM+"23.txt")
    
    return temp

###//2023
def peak_detectedv2(b_l, low, high, fd):
    temp=[]
    prev = b_l[0][0]

    if fd == 1:
        temp[:141] = np.zeros(141)

    else:
    #{'1':low,'2':high-1,'3':fd,'4':"add results of sliding window"}

        # for h in range(0,40):
        for h in range(0,512):
            with open("./data/b_l_range"+str(low)+"_to_"+str(high-1)+".txt"    , "a") as branch_file:

                for w in range(0,141):
                    #TESTER w=5; h=20

                    # if (h==63 or h==62):
                    #     temp.append(2)
    
                    # elif (h%5<2): 
                    if h==0:
                        temp.append(0)

                    elif h==1:
                        temp.append(0)

                    elif (1<h<510):
                        if w <= 3: 
                            temp.append(0)
                        elif (3< w <137):
                
                            ### Starting Peak Detection. Creation of 9x5 sliding window. 
                            th_motion = (
                            b_l[w-4][h-2]+
                            b_l[w-3][h-2]+
                            b_l[w-4][h-1]+
                            b_l[w-3][h-1]+
                            b_l[w-4][h-0]+
                            b_l[w-3][h-0]+
                            b_l[w-4][h+1]+
                            b_l[w-3][h+1]+
                            b_l[w-4][h+2]+
                            b_l[w-3][h+2]+

                            b_l[w-2][h-2]+
                            b_l[w-1][h-2]+
                            b_l[w-0][h-2]+
                            b_l[w+1][h-2]+
                            b_l[w+2][h-2]+

                            b_l[w+4][h-2]+
                            b_l[w+3][h-2]+
                            b_l[w+4][h-1]+
                            b_l[w+3][h-1]+
                            b_l[w+4][h-0]+
                            b_l[w+3][h-0]+
                            b_l[w+4][h+1]+
                            b_l[w+3][h+1]+
                            b_l[w+4][h+2]+
                            b_l[w+3][h+2]+

                            b_l[w-2][h+2]+
                            b_l[w-1][h+2]+
                            b_l[w-0][h+2]+
                            b_l[w+1][h+2]+
                            b_l[w+2][h+2])/30 

                            #print("b_l["    ,w,   "]["   ,h,   "] is ",
                            #      np.round(b_l[w][h],6), " \t 1.5*noise threshold is ", np.round(1.5*th_motion,6))

                            branch_file.write("b_l["  +  str(w)  +  "]["  +  str(h)  +  "] is "  +
                                str(np.round(b_l[w][h],6))  +  " \t 1.5*noise threshold is "  +  str(np.round(1.5*th_motion,6))+"\n")
                            if b_l[w][h]>=1.5*th_motion:
                                
                                # if b_l[w][h] > 2.5*prev:
                                if b_l[w][h] > prev:
                                    branch_file.write("Yes human detected.\n")
                                    temp.append(1)
                                else:
                                    branch_file.write("Move to next.\n")
                                    temp.append(0)
                                prev = b_l[w][h] 
                            else:
                                branch_file.write("Move to next.\n")
                                temp.append(0)
                            
                        
                        else:#(w is 137 138 139 140) #w gt or equal to 137
                            temp.append(2)

                    elif h==510:
                        temp.append(0)

                    elif h==511:
                        temp.append(0)



        
    update_logs091023(low, high, fd, "./data/logs"+DDMM+"23.txt")
    
    return temp

###//2023
def image_show(image,caption):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption=caption)

    with col3:
        st.write(' ')

###Amended from https://doi.org/10.1007/s10877-023-01037-x Acessed date: 06 Oct 23
def VMD(signal, alpha, tau, K, DC, init, tol, low, high, fd):
    # ---------------------
    # signal - the time domain signal (1D) to be decomposed 
    # 
    # alpha - the balancing parameter of the data-fidelity constraint 
    # 
    # tau - time-step of the dual ascent ( pick 0 for noise-slack )
    # 
    # K - the number of modes to be recovered
    # DC - true if the first mode is put and kept at DC (0-freq)
    # 
    # init - 0 = all omegas start at 0
    # 1 = all omegas start uniformly distributed
    # 2 = all omegas initialized randomly
    # tol - tolerance of convergence criterion; typically around 1e-6
    # 
    #
    # Output:
    # -------
    # u - the collection of decomposed modes 
    # u_hat - spectra of the modes 
    # omega - estimated mode center-frequencies 
    #
    # import numpy as np
    # import math
    # import matplotlib.pyplot as plt
    # Period and sampling frequency of input signal
    save_T=len(signal)
    fs=1/float(save_T)
    #print("save_T=:")
    #print(save_T)
    # extend the signal by mirroring 
    # size: 2048 data points
    T=save_T
    f_mirror=np.zeros(2*T)
    f_mirror[0:T//2]=signal[T//2-1::-1]
    f_mirror[T//2:3*T//2]= signal
    f_mirror[3*T//2:2*T]=signal[-1:-T//2-1:-1]
    
    # signal size =2048 data points to f #ab:######## where size of signal was first fixed.
    f=f_mirror
    T=float(len(f))
    t=np.linspace(1/float(T),1,int(T),endpoint=True)
    # Spectral Domain discretization 
    freqs=t-0.5-1/T
    N=2000
    # For future generalizations: individual alpha for each mode
    # penalty index、balance parameter
    Alpha=alpha*np.ones(K,dtype=complex)
    print(TP21) #<<<<<<<<<<<<<<<<<<<         <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<               <<<<<<<<<<<<TP21 start of IMF
    # Construct and center f_hat Fourier transform of signal
    f_hat=np.fft.fftshift(np.fft.fft(f))
    f_hat_plus=f_hat
    f_hat_plus[0:int(int(T)/2)]=0
    
    # matrix keeping track of every iterant
    # could be discarded for mem
    u_hat_plus=np.zeros((N,len(freqs),K),dtype=complex)
    
    #print("u_hat_plus.shape:=", u_hat_plus.shape)
    #print("u_hat_plus:=", u_hat_plus)
    
    # Initialization of omega_k
    omega_plus=np.zeros((N,K),dtype=complex)
    #print("omega_plus.shape:=", omega_plus.shape)
    #print("omega_plus:=", omega_plus)
    
    if (init==1):
        for i in range(1,K+1):
            omega_plus[0,i-1]=(0.5/K)*(i-1)
    elif (init==2):
        omega_plus[0,:]=np.sort(math.exp(math.log(fs))+(math.log(0.5)
        -math.log(fs))*np.random.rand(1,K))
    else:
        omega_plus[0,:]=0
    if (DC):
        omega_plus[0,0]=0
    
    # start with empty dual variables: Lagrange multiplier λ
    lamda_hat=np.zeros((N,len(freqs)),dtype=complex)
    # other inits
    uDiff=tol+2.2204e-16 #updata step 
    n=1 #loop counter
    sum_uk=0 #accumulator
    T=int(T)
    # ----------- Main loop for iterative updates
    # renew n、and minimize u_hat& omega_hat
    # Algorithm 2 Complete optimization of VMD
    
    while uDiff > tol and n<N:
        # while uDiff > tol and n<4:
        # update first mode accumulator
        k=1
        sum_uk = u_hat_plus[n-1,:,K-1] + sum_uk - u_hat_plus[n-1,:,0]
        
        #update spectrum of first mode through Wiener filter of residuals
        # equation in the paper (27)
        freqs_square = np.square(freqs - omega_plus[n-1,k-1]) 
        u_hat_plus[n,:,k-1]=(f_hat_plus - sum_uk\
        - lamda_hat[n-1,:]/2)/(1 + Alpha[k-1] * freqs_square)
        
        # update first omega if not held at 0
        # equation(28) The calculation of the integral is substituted by the inner product.
        if DC==False: 
            omega_plus[n,k-1]= (np.dot(freqs[T//2:T],
            (np.square(np.abs(u_hat_plus[n,T//2:T,k-1]))).T))\
            /(np.sum(np.square(np.abs(u_hat_plus[n,T//2:T,k-1]))))
            #print("n=:",n, "k=:",k, "omega_plus[n,k-1]:=", omega_plus[n,k-1])
        
        for k in range(2,K+1):
            # accumulator
            sum_uk = u_hat_plus[n,:,k-2] + sum_uk - u_hat_plus[n-1,:,k-1]
            
            # mode spectrum 
            u_hat_plus[n,:,k-1]= (f_hat_plus - sum_uk
            - lamda_hat[n-1,:]/2)/(1+Alpha[k-1]
            * np.square(freqs - omega_plus[n-1,k-1]))

            # center frequencies
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],
            np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)\
            /np.sum(np.square(np.abs(u_hat_plus[n,T//2:T:,k-1])))
            
            # Dual ascent
            # equation in the paper(29)
            lamda_hat[n,:]=lamda_hat[n-1,:]\
            +tau*(np.sum(u_hat_plus[n,:,:],axis=1) - f_hat_plus) 

        #loop counter : Comparison with convergence criteria
        n=n+1
        uDiff=2.2204e-16

        for i in range(1,K+1): 
            uDiff=uDiff+1/float(T)*(np.dot(np.conj(u_hat_plus[n-1,:,i-1]
            - u_hat_plus[n-2,:,i-1]).conj().T,
            np.conj(u_hat_plus[n-1,:,i-1]
            - u_hat_plus[n-2,:,i-1]).conj().T)) 
        
        uDiff=np.abs(uDiff)
        #print("n=:", n)
    
    # ------ Postprocessing and cleanup
    # discard empty space if converged early
    N=np.minimum(N,n)
    omega = omega_plus[0:N,:]
    
    # Signal reconstruction. Calculate IMF from frequency components by inverse Fourier transform
    u_hat = np.zeros((T,K),dtype=complex)
    u_hat[T//2:T,:]= np.squeeze(u_hat_plus[N-1,T//2:T,:])
    u_hat[T//2:0:-1,:]=np.squeeze(np.conj(u_hat_plus[N-1,T//2:T,:]))
    u_hat[0,:]=np.conj(u_hat[-1,:])
    u=np.zeros((K,len(t)),dtype=complex)

    for k in range(1,K+1):
        u[k-1,:]= np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k-1])))
    
    # remove mirror part 
    u=u[:,T//4:3*T//4]
    
    for k in range(1,K+1): 
        #filename_ifft = "./data/data_eeg/{}_imf-%d.txt" .format(str(n).zfill(3)) %(k)
        filename_ifft = "./data/data_LFC/{}__imf-%d.txt" .format(str(low).zfill(6)) %k
        np.savetxt(filename_ifft, u[k-1,:])
    
    # recompute spectrum Obtain the frequency spectrum from the IMF by Fourier transform.
    u_hat = np.zeros((T//2, K),dtype=complex)

    for k in range(1,K+1):
        # u_hat[:, k-1]= ((np.fft.fftshift(np.fft.fft(u[k-1,:]))).conj()).T
        u_hat[:, k-1]= ((np.fft.fft(u[k-1,:])).conj()).T
        #print("u_hat.shape=:", u_hat.shape)
    
    fft_amp = np.zeros((T//2, K), dtype = 'float64')
    for k in range(1,K+1):
        filename_fft = "./data/data_LFC/{}__fft-%d.txt" .format(str(low).zfill(6)) %(k)
        np.savetxt(filename_fft, np.abs(u_hat[:,k-1]/ (128 / 2))) 
        fft_amp[:, k-1] = np.abs(u_hat[:,k-1] / (128 / 2))
    
    
    fft_axis = np.linspace(0, 128, int(T/2))
    rcParams['figure.figsize'] = 10,4
    fig = plt.figure()
    
    plt.plot(signal)            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # plt.show()
    filename1 = "./data/data_LFC/sigall.svg"
    fig.savefig(filename1)
    
    
    fig4 = plt.figure()
    N=len(signal)
    f_sig= fftpack.fft(signal) #Amended fft(signal) to fftpack.fft(sig)
    plt.plot(fft_axis[0:int(T/4)], 2.0/N*abs(f_sig[0:int(N/2)])) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    plt.yscale("log")
    # plt.show()
    filename4 = "./data/data_LFC/spectrumall.svg"
    fig4.savefig(filename4)

    for i in range(1,K+1):
        fig2 = plt.figure()
        plt.plot(u[i-1,:])
        plt.title("K is now at %i" %i )
        # plt.show()
        filename2 = "./data/data_LFC/{}__imf-%d.svg" .format(str(low).zfill(6)) %(i)
        fig2.savefig(filename2)
        plt.close(fig2)
    # print("FFT Power Spectrum")
    # for i in range(1,K+1):
    #     fig3 = plt.figure()
    #     plt.plot(fft_axis[0:int(T/4)], fft_amp[0:int(T/4),i-1])
    #     plt.title("K is now at %i" %i )
    #     plt.show() 
    #     filename3 = "./data/data_LFC/{}_fft-%d.svg" .format(str(n).zfill(3)) %(i)
    #     fig3.savefig(filename3)
    plt.close(fig)
    plt.close(fig4)
 
    return (u,u_hat,omega)

###//2023
def range_find(flags_if,dist=5.0):
    perso = 0
    res_list = list(  filter(lambda x: flags_if[x] == 1, range(len(flags_if)))  )
    
    for i in range(len(res_list)):
        res_list[i]=res_list[i]%141

    twist, unique_cnt = np.unique(res_list, return_counts=True)
    tsiwt = dict(zip(twist,unique_cnt))
    print("2) range_find: ",end="")
    print(tsiwt)

    for k in tsiwt:

        if k >= 7 and k <85 and tsiwt.get(k)>110:
            perso = 1
            tarr1_key=k
            # print(k)
            print("b")
        if k >= 85 and tsiwt.get(k)>40:
            perso =+ 1
            # print(k)
            print("a")
            tarr1_key=k
            break
        else:
            tarr1_key=422%141
            print("/",end="")
    #likely range
    dist = (tarr1_key/141)*dist
    print("3) range_find: "+str(dist)+"m\n")

    return perso, tarr1_key

DDMM = "0312"  ### Update Correct Today's Date ###
TP01 = "Program Backend Loaded"
TP02 = "Initialisaton of Pandas DataFrame completed."
TP03 = "Transmit to App"
TP21 = "IMF Save file into data_LPC folder"
TP12 = "Debugging>>>"


def main(argint, arglist):
    
    # read csv from a URL
    setno = argint
    df = pd.read_csv(arglist[setno], header=None)

    ################################################################################################################################################
    ########################################################################
    ########################################################################
    ########################################################################

    now = datetime.datetime.now()
    try:
        f = open("./data/logs"+DDMM+"23.txt","x")
        f.write("Computer Time is " + str(now) + "\n")
        f.close()
        now = datetime.datetime.now()
    except OSError as e:
        print(TP12, " Try deleting files in ./data folder and refresh programme")
        print("\n\nReady, Set, Go!")
        exit()
    else:
        print(TP01," at ",now)

    ### Prepare All My Test Buffers ####Tester:      #df_141complex.shape  #series_141
    df_141i = pd.DataFrame(index=range(len(df.index)),columns=range(141))
    df_141q = pd.DataFrame(index=range(len(df.index)),columns=range(141))
    df_141complex = pd.DataFrame(index=range(len(df.index)),columns=range(141))
    print(TP02)
    #Create sqrt(i^2 + q^2)
    for v in range(141):
        df_141i[v] = df[v] ###+df[i+141]
    for v in range(141):
        df_141q[v] = df[v+141]
    for v in range(141):
        df_141complex[v] = np.sqrt(   np.power(df_141i[v],2) +  np.power(df_141q[v],2)   )

    #Transposed for signal processing
    dataT=df_141complex.transpose()
    ################################################################################################################################################

    ### Prior starting signal processing, the order of windows is chosen as 26 and 50 resp.
    rcParams['figure.figsize'] = 5.7,3.27
    #Window shape Hamming, while the LPF filter
    # freqy above 250Hz/1000 0.25
    numtaps = 26 #page 70:09
    f2 = 0.7/500
    h = signal.firwin(numtaps+1, f2,pass_zero='lowpass',window='hamming')
    #h = signal.firwin(numtaps+1, f2*2,pass_zero='lowpass',window='hamming')
    #plt.stem(fftpack.fftfreq(h.size),     np.abs(fftpack.fft(h)),  linefmt ='--', markerfmt ='')

    numtaps = 50 #page 70:09
    f2 = 0.7/1000
    h50 = signal.firwin(numtaps+1, f2*2)
    #plt.xlim(0,0.6)

    ###Use of LPF n Smooth Filter  (17 Sep 23)
    s = (len(df_141complex[0]),141)

    array1=np.zeros(s, dtype=complex)

    array2=np.zeros(s, dtype=complex)

    M = 50
    coef = np.hamming(M+1)
    #coef = signal.windows.hamming(M+1)
    start_dist_of_interest = 0



    ### Starting to process DataT
    for i in range(0,len(df_141complex[0])):  
        filtered_24 = signal.lfilter(h,1,dataT[start_dist_of_interest]) 
        topl=np.copy(filtered_24[:12])
        for i in range(0,141):
            if i<129: #0 to 128
                filtered_24[i]=filtered_24[i+12]
            else:     #129 130 131 132 133 4 5 6 7 8 9 140
                filtered_24[i]=topl[i-129]
        temp=fftpack.fft(filtered_24)
        filtered_24b = fftpack.ifft(temp) #you'll want to print this for proof #add into 2D array
        
        ######################
        #Cascaded a second filter
        # dataT_sfil= np.convolve(fftpack.ifft(temp),coef,mode='same')           #add into 2D arr
        dataT_sfil = signal.lfilter(h50,1,filtered_24b)
        topr=np.copy(dataT_sfil[:25])
        for i in range(0,141):
            if i<116: #0 to 115
                dataT_sfil[i]=dataT_sfil[i+24]
            else:     #116 117 ... 131 132 133 4 5 6 7 8 9 140
                dataT_sfil[i]=topr[i-116]
        
        array1[start_dist_of_interest]=filtered_24b
        array2[start_dist_of_interest]=dataT_sfil
        
        start_dist_of_interest=start_dist_of_interest+1 #until 0,1131-1,1131

    print('Done Signal Proc, 3 charts will be displayed.')

    ################################################################################################################################################

    ### Starting Background Noise Removal
    beta=0.97 #page 70:10 #𝑐𝑘(𝑡) = 𝛽𝑐𝑘−1(𝑡) + (1 − 𝛽)𝑟𝑘(𝑡)
    k=7

    s = (len(df_141complex[0]),len(array2[0]))
    B2=np.zeros(s)
    clutter_suppr=np.zeros(s)

    for k in range(0, len(df_141complex[0])):
        for i in range(0, len(dataT[0])):
            if i == 0:
                B2[k][i] = (1-beta)*np.abs(array2[k][i])
            else: 
                B2[k][i] = (1-beta)*np.abs(array2[k][i]) + beta*B2[k][i-1]
            
            clutter_suppr[k][i] = np.abs(array2[k][i]) - B2[k][i]
    print("Done Clutter Shift, 2nd heatmap in line will be displayed.")

    ################################################################################################################################################

    ### Starting User Identification

    #output b_s and b and comparision statement
    ###Total: 1131 | Strategically Take x[0-9][0] | Take x[0-9][1] | ...| Take x[0-9][140]
    ###Created by the above strategic, a 9x141 short observation. There r 4 short in 1 long observ
    s = (141,64); b_s=np.zeros(s) #16 #10  #64
    s = (141,512); b_l=np.zeros(s) #64 #40  #512
    s = (64); tem=np.zeros(s)
    s = (512); longobsvtem=np.zeros(s)
    c = 1.2 #page 70:10
    cnt = 0

    #Update fastdetected database, How to use:
    #Mark the arbituary outcome of fast movement >Take the 'low' indices > Put
    #inside a queue >This in turn will be used by PEAK DETECTIONv2
    fastdetected = 0

    while(cnt<np.floor(len(df_141complex[0])/512)):
        # Used once
        perso = 0
        ####For the 1st b_l block... get b_l block for [0-40)
        low=512*cnt; high=low+512 #512 will be exclusive
        ptr = np.random.randint(low,high-63)            #randomise any 10 data for bs. Param 2 exclusive.
        for j in range(0,141):
            for i in range(ptr,ptr+64):      #any 100 between [0-30] 30 incluc.
                
                tem[i-ptr]=clutter_suppr[i][j]      #clutter_suppr[10][0] to clutter_suppr[19][0]
            
            b_s[j][:]=np.abs(fftpack.fft(tem))
        print('TP11: ptr is ', ptr, ' cnt is ', cnt)

        cnt=cnt+1
        ########################################################

        #For 1x single b_l block, finding out FFT is useable or not useable
        signalSTR = np.max(b_s)/(np.sum(b_s)/(64*141))

        for j in range(0,141):
            for i in range(low,high):      #  [40-80)
                
                longobsvtem[i-low]=clutter_suppr[i][j]      #clutter_suppr[40][0] to clutter_suppr[79][0]
                                                        #longobsvtem[40]
                                                        #i = [0:40)
                                                        #i = [1080:1120)
                                                        #i = [1120:~not computerised]
            
            b_l[j][:]=np.abs(fftpack.fft(longobsvtem))

        
        signalSTRL = np.max(b_l)/(np.sum(b_l)/(512*141))
        if signalSTR>(c*signalSTRL):
            print(signalSTR,signalSTRL*c,'To Be Discarded.') #page 70:10
            fastdetected = 1
            
        else:
            print(signalSTR,signalSTRL*c,'No Fast Movement Detected, Proceed.')
            fastdetected = 0

        ############################################################################################################################################
        
        #Continued, so if FFT is useable, find peak amplitude

        flags_if = peak_detectedv2(b_l, low, high, fastdetected)
        # 1) visualisation for console print-out
        for i in range(len(flags_if)):
            if i%141 == 140:
                print(flags_if[i])
            else:
                print(flags_if[i],end="")
        # 2) Peaks Detected in range_find()
        # 3) Likely Range in range_find()
        perso, tarr1_key = range_find(flags_if)
        

        # print("CheckPoint: "+arglist[setno]+" | Subject: "+str(perso)+"\n\n")
        print("\n\n")
        print("Done Selecting Peaks, Logs Available. ")

    ################################################################################################################################################

        #### Starting "Variational Mode Decomposition" -- VMD Algorithm accessed 18 Oct 23

        alpha = 2000.0 
        tau = 0 
        DC = 0 
        init = 1 
        tol = 1e-7


        K = 3 #6

        tarr1 = np.transpose(clutter_suppr[low:high])   #tarr1 is used once
        # u,u_hat,omega = VMD(tarr1[tarr1_key], alpha, tau, K, DC, init, tol, low, high, fastdetected)
        VMD(tarr1[tarr1_key], alpha, tau, K, DC, init, tol, low, high, fastdetected)
    
    ########################################################################
    ########################################################################
    ########################################################################
    ################################################################################################################################################

    return dataT, df_141complex, array1, array2, perso, clutter_suppr