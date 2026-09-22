using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Mvc.Rendering;

namespace ChequeVerification.Web.ViewModels.ReferenceSignatures;

public class ReferenceSignatureCreateViewModel
{
    [Required]
    [Range(1, int.MaxValue, ErrorMessage = "Client invalide.")]
    public int CustomerId { get; set; }

    public string CustomerNumber { get; set; } = string.Empty;

    public string CustomerFullName { get; set; } = string.Empty;

    public int ActiveReferenceCount { get; set; }

    public int RecommendedTarget { get; set; } = 5;

    [Display(Name = "Image de la signature de référence")]
    [Required(ErrorMessage = "L'image de la signature est obligatoire.")]
    public IFormFile? ImageFile { get; set; }
}
