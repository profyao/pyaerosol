#include<stdio.h>

void c_interpol_2d(double * interpol, double tau, double * lookup, int * lookup_size){

    /*lookup point to lookup_size[0]*n array*/

    double tau_grid[13]={0,0.05,0.1,0.2,0.4,0.6,0.8,1,1.5,2,3,4,6};
    double yy1, yy2, xx1=0, xx2=6;
    int    ini=0, inj, pos;

    if (tau<=0){
        interpol[0] = -1; /*error message*/
        printf("tau == %f \n", tau);
    }

    else if (tau>6){
        for (ini = 0; ini<lookup_size[1]; ini++)
            interpol[ini] = *(lookup+lookup_size[0]-1+ini*lookup_size[0]);
    }

    else {

        xx1 = tau_grid[0];  /*left bracket*/
        inj = 1;

        while (!((xx1<(tau)) && (tau<=tau_grid[inj])))
            xx1 = tau_grid[inj++];

        xx2 = tau_grid[inj]; /*right bracket*/

        for (ini = 0; ini < lookup_size[1]; ini++){
            pos = ini*13;
            yy1 = *(lookup+pos+inj-1);
            yy2 = *(lookup+pos+inj);
            interpol[ini] = yy1 + (tau-xx1)*(yy2-yy1)/(xx2-xx1);
        }
    }

}

void c_interpol_3d(double * interpol, double tau, double * lookup, int * lookup_size){

	double tau_grid[13]={0,0.05,0.1,0.2,0.4,0.6,0.8,1,1.5,2,3,4,6};
	double yy1, yy2, xx1=0, xx2=6, zz1, zz2;
	int    ini=0, inj, ink=0, pos, jj;

    if (tau<=0){
        interpol[0] = -1; /*error message*/
        printf("Error: tau == %f \n", tau);
    }

    else if (tau>6){

        for (ink = 0; ink<lookup_size[2]; ink++)
            for (ini = 0; ini<lookup_size[1]; ini++)
                interpol[ini+ink*lookup_size[1]] = *(lookup+(lookup_size[0]-1)+ini*lookup_size[0]+ink*lookup_size[0]*lookup_size[1]);

    }

    else{

        inj = 0;
        while (tau>tau_grid[++inj]);

        inj--;
        xx1 = tau_grid[inj];        /*left bracket*/
        xx2 = tau_grid[inj+1];      /*right bracket*/
        switch (inj){
            case 0:
            case 1:
            case 2:
            case 3:
            case 9:
            case 10:
            case 11: /*linear interpolation*/
                    for (ink = 0; ink < lookup_size[2]; ink++)
                        for (ini = 0; ini < lookup_size[1]; ini++){
                            pos = ini*lookup_size[0]+ink*lookup_size[0]*lookup_size[1];
                            yy1 = *(lookup+pos+inj);
                            yy2 = *(lookup+pos+inj+1);
                            interpol[ini+ink*lookup_size[1]] = yy1 + (tau-xx1)*(yy2-yy1)/(xx2-xx1);   /*need to check!!!*/
                        }
                    break;
            case 4:
            case 5:
            case 6:
            case 7:
            case 8: /*quadratic interpolation*/
                    for (ink = 0; ink < lookup_size[2]; ink++)
                        for (ini = 0; ini < lookup_size[1]; ini++){
                            pos = ini*lookup_size[0]+ink*lookup_size[0]*lookup_size[1];
                            /*calculate all z_j values, j=4,...,inj+1 */

                            zz1 = (*(lookup+pos+4)-*(lookup+pos+3))/(0.4-0.2);
                            zz2 = -zz1 + 2*(*(lookup+pos+5)-*(lookup+pos+4))/(0.6-0.4);

                            for (jj = 5; jj < inj+1; jj++){
                                zz1 = zz2;
                                zz2 = -zz1 + 2*(*(lookup+pos+jj+1)-*(lookup+pos+jj))/(tau_grid[jj+1]-tau_grid[jj]);
                            }

                            yy1 = *(lookup+pos+inj);
                            interpol[ini+ink*lookup_size[1]] = yy1 + zz1*(tau-xx1) + (tau-xx1)*(tau-xx1)*(zz2-zz1)/(2*(xx2-xx1)); /*need to check!!!*/
                        }
                    break;
        }
    }
}
