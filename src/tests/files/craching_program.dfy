// Insertion sort.
//
// Author: Snorri Agnarsson, snorri@hi.is

// dafny_fork/Binaries/Dafny verify craching_program.dfy --cores 1 --verification-time-limit 300

predicate IsSorted( s: seq<int> )
{
    forall p,q | 0<=p<q<|s| :: s[p]<=s[q]
}
method InsertionSort( s: seq<int> ) returns ( r: seq<int> )
    ensures multiset(r) == multiset(s);
    ensures IsSorted(r);
{
    r := [];
    var rest := s;
    while rest != []
        decreases rest;
        invariant multiset(s) == multiset(r)+multiset(rest);
        invariant IsSorted(r);
    {
        var x := rest[0];
        rest := rest[1..];
        var k := |r|;
        while k>0 && r[k-1]>x
            invariant 0 <= k <= |r|;
            invariant forall p | k<=p<|r| :: r[p]>x;
        {
            k := k-1;
        }
        assert r == r[..k]+r[k..];
        assert multiset(r[..k]) + multiset(r[k..]) == multiset(r);
        r := r[..k]+[x]+r[k..];
    }
}
