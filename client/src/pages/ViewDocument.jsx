import React, { useState } from 'react'
import { useParams } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import { ChevronLeft, FileText, ArrowDownToLine, Trash } from 'lucide-react';

const ViewDocument = () => {
    const { id } = useParams();
    const { toast, myDocuments, navigate, axios } = useAppContext();
    const [showConfirm, setShowConfirm] = useState(false);
    const currentDocuments = myDocuments.find((item) => item._id === id);

    const deleteDocument = async () => {
        toast.loading("Document deleting...");
        try {
            const response = await axios.delete('/api/document/delete', {
                data: { id },
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            if (response.data.success) {
                toast.dismiss()
                toast.success('Document deleted successfully');
                navigate('/my-reports', scrollTo(0,0));
            } else {
                toast.dismiss()
                toast.error('Error:', response.data.message);
            }
        } catch (error) {
            toast.dismiss()
            toast.error('Delete request failed:', error.response?.data || error.message);
        }
    };


    const downloadDocument = async () => {
        if (
            !currentDocuments ||
            !currentDocuments.image ||
            currentDocuments.image.length === 0
        ) {
            toast.error("No document available for download.");
            return;
        }

        const fileUrl = currentDocuments.image[0];

        try {
            const response = await fetch(fileUrl, { mode: 'cors' });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const blob = await response.blob();
            const blobUrl = window.URL.createObjectURL(blob);

            // Extract the extension from the URL (e.g., .jpg, .pdf)
            const extension = fileUrl.split('.').pop().split('?')[0];
            const filename = `${currentDocuments.name}.${extension}`;

            const link = document.createElement("a");
            link.href = blobUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            window.URL.revokeObjectURL(blobUrl);
        } catch (error) {
            toast.error("Download failed:", error);
        }
    };



    return (
        <div className='md:p-10 py-6'>
            <button className='fixed m-1 w-10 h-10 shadow bg-slate-50 text-gray-600 rounded-full flex justify-center items-center cursor-pointer transition-transform duration-300 ease-in-out transform hover:scale-105' onClick={() => navigate('/my-reports')}><ChevronLeft size={20} /></button>
            {
                currentDocuments ? (
                    <div className='flex flex-col items-center justify-center gap-8 px-10'>
                        <div>
                            {currentDocuments.image && currentDocuments.image.length > 0 && (() => {
                                const fileUrl = currentDocuments.image[0];
                                const isPdf = fileUrl.toLowerCase().endsWith('.pdf');

                                return isPdf ? (

                                    <object data={fileUrl} type="application/pdf" className='w-full p-10 flex flex-col justify-center items-center gap-2'>
                                        <FileText size={32} className="text-red-500" />
                                        <p>Your browser does not support PDFs..</p>
                                    </object>

                                ) : (
                                    <img
                                        src={fileUrl}
                                        alt="Document Preview"
                                        className="md:h-auto md:w-[50%] h-[80%] mx-auto  object-contain"
                                    />
                                );


                            })()}
                        </div>

                        <div className='w-full sm:w-1/2 flex flex-row items-center justify-between px-6 sm:gap-40 bg-slate-50 rounded-xl p-4 border border-gray-200'>
                            <div className='flex flex-col items-start gap-2 text-nowrap'>
                                <h1 className='font-medium'>{currentDocuments.name}</h1>
                                <p className='text-xs'>Date: {new Date(currentDocuments.createdAt).toLocaleDateString()}</p>
                                <p>Report Result: {}</p>
                            </div>
                            <div className='flex gap-4'>
                                <button onClick={() => setShowConfirm(true)} className='w-10 h-10 bg-special-red rounded-full flex justify-center items-center transition-transform duration-300 ease-in-out transform hover:scale-105 cursor-pointer'><Trash size={22} color='white' /></button>
                                {showConfirm && (
                                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#00000066]">
                                        <div className="bg-white rounded-xl p-6 shadow-xl m-6 w-full max-w-md">
                                            <h2 className="text-lg font-semibold mb-4">Confirm Delete</h2>
                                            <p className="mb-6">Are you sure you want to delete this document?</p>
                                            <div className="flex justify-end space-x-4">
                                                <button
                                                    onClick={() => setShowConfirm(false)}
                                                    className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 cursor-pointer"
                                                >
                                                    Cancel
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        deleteDocument(),
                                                            setShowConfirm(false);
                                                    }}
                                                    className="px-4 py-2 bg-special-red/90 text-white rounded hover:bg-special-red cursor-pointer"
                                                >
                                                    Delete
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <button onClick={downloadDocument} className="w-10 h-10 bg-green-600 rounded-full flex justify-center items-center transition-transform duration-300 ease-in-out transform hover:scale-105 cursor-pointer">
                                    <ArrowDownToLine size={22} color="white" />
                                </button>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className='flex flex-col items-center justify-center gap-4 sm:gap-10 p-4'>
                        <div className='h-screen w-full sm:w-1/2 rounded-lg'>
                        </div>
                        <div className=' w-full sm:w-auto flex flex-row items-center justify-between sm:gap-40'>
                            <div className='flex flex-col items-start gap-2 '>
                                <div className='h-10 w-60 rounded-lg'>
                                </div>
                                <div className='h-10 w-40 rounded-lg'>
                                </div>
                            </div>

                            <div className='h-18 w-18 rounded-full'>
                            </div>
                        </div>

                    </div>
                )
            }
        </div>
    )
}

export default ViewDocument
