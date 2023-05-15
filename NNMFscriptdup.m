%uiimport('-ui import');
A=table2array(testfile);

[W2,H2]=nnmf(A, 2, 'replicates', 50, 'algorithm', 'als');
[W3,H3]=nnmf(A, 3, 'replicates', 50, 'algorithm', 'als');
[W4,H4]=nnmf(A, 4, 'replicates', 50, 'algorithm', 'als');
[W5,H5]=nnmf(A, 5, 'replicates', 50, 'algorithm', 'als');
[W6,H6]=nnmf(A, 6, 'replicates', 50, 'algorithm', 'als');
[W7,H7]=nnmf(A, 7, 'replicates', 50, 'algorithm', 'als');

C2=W2*H2;
C3=W3*H3;
C4=W4*H4;
C5=W5*H5;
C6=W6*H6;
C7=W7*H7;

R2All(1) =corr(C2(:),A(:))^2;
R2All(2) =corr(C3(:),A(:))^2;
R2All(3) =corr(C4(:),A(:))^2;
R2All(4) =corr(C5(:),A(:))^2;
R2All(5) =corr(C6(:),A(:))^2;
R2All(6) =corr(C7(:),A(:))^2;

X=[2 3 4 5 6 7];
plot(X,R2All);