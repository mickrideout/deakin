GM <- function(x) { prod(x)^(1/length(x)) }
HM = function(x) {
  if(prod(x)==0) {
    return (0)
  } else {
    length(x)/sum(1/x)
  }
}
