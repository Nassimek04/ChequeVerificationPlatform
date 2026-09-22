using ChequeVerification.Web.Models.Entities;

namespace ChequeVerification.Web.ViewModels.Customers;

public class CustomerDetailViewModel
{
    public Customer Customer { get; set; } = null!;
    public int ChequeCount { get; set; }
    public int ReferenceSignatureCount { get; set; }
    public List<Cheque> Cheques { get; set; } = new();
    public List<ReferenceSignature> ReferenceSignatures { get; set; } = new();
}