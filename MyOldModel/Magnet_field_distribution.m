clear
clc
mu0=4*pi*10^-7;
mur=1.03;
Br=1.22;
alpha_r=0.9;
p=6;
k=30;
Hm=4;
g=8;
Rs=18;
L=140
Rm=Rs+g;
Rr=Rs+Hm+g;
Ba=zeros();
B=zeros(360,1);
M_=4*Br/(pi*mu0);
alpha_=pi*alpha_r/2;
%Calculation Fourier series of Magnet induction 
for i=1:k
  n=2*i-1;
  np=n*p;
  M=M_*sin(alpha_*n)/n;
  if (i==1)
    Ba(i)=(mu0*M/mur)*((Rm/Rs)^(2)-(Rr/Rs)^(2)+(Rr/Rs)^(2)*log((Rm/Rr)^(2)))/(((mur+1)/mur)*(1-(Rr/Rs)^(2))-((mur-1)/mur)*((Rm/Rs)^2-(Rr/Rm)^2));
    else  
    Ba(i)=2*(mu0*M/mur)*(np/(np^2-1))*((Rs/Rm)^(np+1))*((np-1)+2*(Rr/Rm)^(np+1)-(np+1)*(Rr/Rm)^(2*np))/(((mur+1)/mur)*(1-(Rr/Rs)^(2*np))-((mur-1)/mur)*((Rm/Rs)^(2*np)-(Rr/Rm)^(2*np)));
    end
  end
  plot(Ba)
for ang_=1:360
  tetta=ang_/360*(2*pi);
  for i=1:k
    n=2*i-1;
    B(ang_)=B(ang_)+Ba(i)*cos(n*p*tetta);
    end
  end
  plot(B)
Be=zeros(0);  
for i=1:360
  Be_=0;
  for k=1:360/(2*p)
    if i+k<=360
    Be_=B(i+k)+Be_;
    else
    Be_=B(i+k-360)+Be_;
    end
    end
  Be(i)=Be_;
  end
  Be=Be*2*p/(360);
  psi=Be*L*2*3.14*Rs/(1000*1000);

  e=psi*314;
  plot(e)   
  w=round(24/max(e))
  
%B=2*(mu0*M(n)/mur)*(n*p/((n*p)^2-1))*(Rs/Rm)^(n*p-1)*((n*p-1)*(Rm)^(2*n*p)+2*(Rr)^(2*n*p+1)*(Rm)^(2*n*p-1)-(n*p-1)*(Rr)^(2*n*p))/(((mur+1)/mur)*((Rs)^(2*n*p)-(Rm)^(2*n*p))-((mur-1)/mur)*((Rm)^(2*n*p))-(Rs*Rr/Rm)^(2*n*p))