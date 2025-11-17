nst=24;
mu0=4*pi*10^-7;
m=3;
p=4;
b0=0.5;
Hm=4;
g=8;
Rs=18;
Rm=Rs+g;
Rr=Rs+Hm+g;
q=nst/(2*p*m);
sov=b0/(2*Rs);
dv=pi/nst;
r=Rs*0.001;
Rs=Rs*0.001;
Rr=Rr*0.001;
y=pi/p;
w=12;
u_=[1,5,7,11,13,17,19,23];
Iu=[10,1,0.2,0,0,0,0,0];
Qu=[0,0,0,0,0,0,0,0];
wr=314;
Bwind=zeros();
hold on
for i=1:1
  t=0.1*(i-1);
  for i=1:360
  ang_=i*pi/180;
  M=0;
  for k=1:8
    u=u_(k);
    X=0;
      for c=1:100
 %       c=c-1;
 %       if (c=0)
%          v=p*(6*c+u)
          v=c;
 %         Ksov=sin(v*sov)/(v*sov);
 %         Fv=(g+Hm)*(v/r)*((r/Rs)^v)*(1+(Rr/r)^(2*v))/(1-(Rr/Rs)^(2*v));
 %         Kdv=sin(q*v*dv)/(q*sin(v*dv));
 %         Kpv=sin(v*y/2);
  %        X=X+Ksov*Fv*Kdv*Kpv*sin(u*p*wr*t-v*ang_+Qu(k))/v;
         % X=X+Ksov*Fv*Kdv*Kpv*sin(u*p*wr*t+v*ang_+Qu(k))/v;
  %        else
  %        v=p*(6*c-u)
   %       Ksov=sin(v*sov)/(v*sov);
   %       Fv=(g+Hm)*(v/r)*((r/Rs)^v)*(1+(Rr/r)^v)/(1-(Rr/Rs)^v);
    %      Kdv=sin(q*v*dv)/(q*sin(v*dv));
    %      Kpv=sin(v*y);
    %      X=X+Ksov*Fv*Kdv*Kpv*sin(u*p*wr*t+v*ang_+Qu(k))/v;
     %     v=p*(6*c+u);
          Ksov=sin(v*sov)/(v*sov);
          Fv=(g+Hm)*(v/r)*((r/Rs)^v)*(1+(Rr/r)^(2*v))/(1-(Rr/Rs)^(2*v));
          Kdv=sin(q*v*dv)/(q*sin(v*dv));
          Kpv=sin(v*y);
          X=X+Ksov*Fv*Kdv*Kpv*(sin(u*(p*wr*t)+Qu(k))*cos(v*ang_)+sin(u*(p*wr*t-2*pi/3)+Qu(k))*cos(v*ang_-2*pi/3)+sin(u*(p*wr*t-4*pi/3)+Qu(k))*cos(v*ang_-4*pi/3))/v;
         end
      M=M+X*Iu(k);  
      end
    Bwind(i)=M*mu0*w/(pi*(g+Hm));
    end
    plot(Bwind)
  end

plot(Bwind)