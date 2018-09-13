wilson.conf = function(phat, n, z){
  a = (phat + z*z/(2*n))/ (1 + z*z/n)
  b = (z/(1 + z*z/n))* sqrt((phat*(1-phat)/n)+(z*z/(4*n*n)))
  return(a + b*c(-1,1))
}