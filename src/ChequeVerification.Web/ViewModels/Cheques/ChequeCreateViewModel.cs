using Microsoft.AspNetCore.Mvc.Rendering;
using System.ComponentModel.DataAnnotations;

namespace ChequeVerification.Web.ViewModels.Cheques;

public class ChequeCreateViewModel
{
    [Display(Name = "Client")]
    [Range(1, int.MaxValue, ErrorMessage = "Sélectionnez un client.")]
    public int CustomerId { get; set; }

    public IEnumerable<SelectListItem> Customers { get; set; } = new List<SelectListItem>();

    [Display(Name = "Numéro du chèque")]
    public string ChequeNumber { get; set; } = string.Empty;

    [Display(Name = "Montant (MAD)")]
    [Range(0, 999999999, ErrorMessage = "Montant invalide.")]
    public decimal? Amount { get; set; }

    [Display(Name = "Date d'émission")]
    [DataType(DataType.Date)]
    public DateOnly? IssueDate { get; set; }

    [Display(Name = "Image du chèque")]
    [Required(ErrorMessage = "L'image du chèque est obligatoire.")]
    public IFormFile? ImageFile { get; set; }
}